# -*- coding: utf-8 -*-
# @Time : 2022/12/6 上午9:44
# @Author : YANG.C
# @File : train.py
import copy
import random
import time
import math
import warnings
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import classification.datasets
from classification.datasets.builder import build_dataset, build_dataloader
from classification.models.builder import CLASSIFIERS
from classification.models.classifiers import Classifier
from classification.solver.optimizer import build_optimizer
from classification.solver.lr_scheduler import warpper_lr_scheduler
from classification.utils.dist_utils import init_seeds, de_parallel
from classification.utils.dist_utils import get_dist_info
from utils.log import logger
from classification.api.evaluate import evaluate, evaluate_mulitlabel, evaluate_splitfc, evaluate_fairness
from classification.utils.utils import profile, load_resume_model, load_checkpoints
from classification.utils.ema import ModelEMA
from classification.models.backbones.mobileone import reparameterize_model

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train_model(cfg):
    # basic settings
    init_seeds(1 + RANK)
    distributed = True if RANK != -1 else False
    cfg['WORLD_SIZE'] = WORLD_SIZE

    # logger
    if RANK in [0, -1]:
        for k, v in dict(cfg).items():
            logger.debug(f'{k}: {v}')

    # cuda setting
    cuda = torch.cuda.is_available()
    if cuda and LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    if cuda and LOCAL_RANK == -1:
        device = torch.device('cuda', 0)

    # config from default_runtime.py
    log_period = cfg.log_periods

    # build model
    model_config = cfg.model.copy()
    model_type = model_config.pop('type')
    model = CLASSIFIERS[model_type](model_config)

    # for kd
    teacher_model = None
    teacher_target = None
    if 'distillation' in cfg and cfg['distillation'] is True:
        teacher_model_config = cfg.model_teacher.copy()
        teacher_model_type = teacher_model_config.pop('type')
        teacher_model = CLASSIFIERS[teacher_model_type](teacher_model_config)
        teacher_model = load_checkpoints(teacher_model, teacher_model_config['ckpt'])       # overwrite
        teacher_model.eval()
        logger.debug(f'distillation mode, load teacher model successful')

    use_ema = True
    ema_model = copy.deepcopy(model)
    ema = ModelEMA(ema_model)

    # forward test
    if RANK in [-1, 0]:
        model.eval()
        inputs = torch.randn((1, 3, 224, 224))
        gt_labels = torch.LongTensor([0])
        outputs = model(inputs, gt_labels)
        logger.debug(f'testing case!!!: input: {inputs.shape}, output: {outputs}')
        tb_writer = SummaryWriter(log_dir=cfg.tensorboard_dir)
        profile(inputs, model, n=30)  # print

    if cuda:
        model.cuda()
        if 'distillation' in cfg and cfg['distillation'] is True:
            teacher_model.cuda()

    # datasets
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)

    # data loaders
    loader_cfg = dict(
        num_gpus=cfg['WORLD_SIZE'],
        dist=distributed,
        round_up=True,
        seed=42,
        sampler_cfg=cfg.get('sampler', None),
    )
    loader_cfg.update(
        {k: v for k, v in cfg.data.items() if k not in ['train', 'val', 'test']}
    )

    train_loader_cfg = {**loader_cfg}
    train_loader = build_dataloader(train_dataset, **train_loader_cfg)
    if RANK in [-1, 0]:
        loader_cfg['num_gpus'] = 1
        loader_cfg['dist'] = False
        loader_cfg['samples_per_gpu'] *= 64
        val_loader_cfg = {**loader_cfg}
        val_loader_cfg['collect'] = False
        val_loader = build_dataloader(val_dataset, **val_loader_cfg)

    # scheduler and optimizer
    nb = len(train_loader)
    epochs = cfg.max_epochs
    optimizer = build_optimizer(cfg.optimizer, model)
    scheduler_cfg = {
        'steps_per_epoch': nb,
        'epochs': epochs,
    }
    # scheduler = warpper_lr_scheduler(cfg.lr_scheduler, optimizer, scheduler_cfg)
    scheduler = lr_scheduler.OneCycleLR(optimizer, cfg.optimizer.lr, total_steps=None, epochs=epochs,
                                        steps_per_epoch=nb, pct_start=0.1)
    optimizer.zero_grad()

    # DDP
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=[LOCAL_RANK], find_unused_parameters=True,
                    broadcast_buffers=False)
        if 'distillation' in cfg and cfg['distillation'] is True:
            teacher_model = DDP(teacher_model, device_ids=[LOCAL_RANK], output_device=[LOCAL_RANK],
                                find_unused_parameters=True,
                                broadcast_buffers=False)

    # Start training
    t0 = time.time()
    start_epoch = 0
    best_acc_score = 0
    best_fairness_score = 0
    best_performance_score = 0
    best_epoch = 0
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)
    accumulate_step = 0

    if 'distillation' in cfg and cfg['distillation'] is True:
        logger.debug('distillation mode, ready to evaluate teacher model...')
        if RANK in [-1, 0]:
            cfg.device = device
            torch.cuda.empty_cache()
            with torch.no_grad():
                result = evaluate_fairness(val_loader, teacher_model, cfg)
                acc_score = result['acc_score']
                overall_score = result['overall_score']
                fairness_score = result['fairness_score']
                performance_score = result['performance_score']
                logger.debug(f'teacher model, acc_socre: {acc_score}, fairness_score: {fairness_score}, '
                             f'performance_socre: {performance_score}')

    for epoch in range(start_epoch, epochs):
        model.train()
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        pbar = enumerate(train_loader)
        optimizer.zero_grad()

        for i, batch in pbar:
            # for debugging validation
            # if i > 50:
            #     break
            ni = i + nb * epoch
            imgs = batch['image']
            imgs = imgs.to(device, non_blocking=True).float()

            if 'multi_scale' in cfg and cfg['multi_scale'] and random.uniform(0, 1) < cfg['multi_scale_proc']:
                gs = 32 if 'down_sample_size' not in cfg else cfg['down_sample_size']
                ms_min = 0.5 if 'scale_ratio' not in cfg else cfg['scale_ratio'][0]
                ms_max = 1.5 if 'scale_ratio' not in cfg else cfg['scale_ratio'][1]
                sz = random.uniform(ms_min * cfg['imgsz'], ms_max * cfg['imgsz'] + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(s * sf / gs) * gs for s in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            targets = batch['gt_label']
            targets = targets.to(device)

            with amp.autocast(enabled=cuda):
                if 'distillation' in cfg and cfg['distillation'] is True:
                    teacher_target = teacher_model(imgs)['classes']
                    loss, losses = model(imgs, targets, teacher_target)
                else:
                    loss, losses = model(imgs, targets)

            # backward
            scaler.scale(loss).backward()

            # optimizer
            if ni - last_opt_step >= accumulate_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # scheduler.scheduler_iteration_hook()
                scheduler.step()

                # update ema
                ema.update(model)

            if RANK in [-1, 0] and i % log_period == 0:
                lr = [x['lr'] for x in optimizer.param_groups]
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                logger.info(f'[{epoch}/{epochs - 1}]' + f'[{i}/{nb}]' +
                            f'GPU: {mem}' + '\t' + f'imgsz: {imgs.shape[2]}' + '\t' + f'triplet loss: {losses[0]:.4}' +
                            '\t' + f'arcface loss: {losses[1]:.4}' + '\t' + f'kd loss: {losses[2]:.4}' + '\t' +
                            f'loss: {loss:.4}' + '\t' + f'LR: {lr[0]:.4}')
                tb_writer.add_scalar('total loss', loss.cpu().item(), ni)
                tb_writer.add_scalar('lr', lr[0], ni)

        if RANK in [-1, 0]:
            cfg.device = device
            torch.cuda.empty_cache()
            with torch.no_grad():
                result = evaluate_fairness(val_loader, model, cfg)
                acc_score = result['acc_score']
                overall_score = result['overall_score']
                fairness_score = result['fairness_score']
                performance_score = result['performance_score']

                # test ema
                ema_result = evaluate_fairness(val_loader, copy.deepcopy(ema.ema).cuda(), cfg)
                ema_acc_score = ema_result['acc_score']
                ema_overall_score = ema_result['overall_score']
                ema_fairness_score = ema_result['fairness_score']
                ema_performance_score = ema_result['performance_score']

                ckpt = {
                    'epoch': epoch,
                    'acc_score': acc_score,
                    'fairness_score': fairness_score,
                    'performance_score': performance_score,
                    'model': copy.deepcopy(de_parallel(model)),
                    'ema': copy.deepcopy(de_parallel(ema.ema)) if use_ema else None,
                    'updates': None,
                    'optimizer': None,
                }
                if max(acc_score, ema_acc_score) > best_acc_score:
                    best_acc_score = max(acc_score, ema_acc_score)
                    torch.save(ckpt, f'{cfg.checkpoint_dir}/best_acc_e{epoch}_{max(acc_score, ema_acc_score):.3f}.pt')
                    torch.save(ckpt, f'{cfg.checkpoint_dir}/best_acc.pt')
                if max(fairness_score, ema_fairness_score) > best_fairness_score:
                    best_fairness_score = max(fairness_score, ema_fairness_score)
                    torch.save(ckpt,
                               f'{cfg.checkpoint_dir}/best_fair_e{epoch}_{max(fairness_score, ema_fairness_score):.3f}.pt')
                    torch.save(ckpt, f'{cfg.checkpoint_dir}/best_fair.pt')
                if max(performance_score, ema_performance_score) > best_performance_score:
                    best_performance_score = max(performance_score, ema_performance_score)
                    best_epoch = epoch
                    torch.save(ckpt,
                               f'{cfg.checkpoint_dir}/best_perf_e{epoch}_{max(performance_score, ema_performance_score):.3f}')
                    torch.save(ckpt, f'{cfg.checkpoint_dir}/best_perf.pt')

                logger.info(
                    f'epoch: {epoch}, overall_score: {overall_score}, acc_score: {acc_score}, fairness_score: {fairness_score}, performance_score: {performance_score}, best_performance_score: {best_performance_score}, best_epoch: {best_epoch}')
                logger.info(
                    f'epoch: {epoch},EMA overall_score: {ema_overall_score}, acc_score: {ema_acc_score}, fairness_score: {ema_fairness_score}, performance_score: {ema_performance_score}, best_performance_score: {best_performance_score}, best_epoch: {best_epoch}')

    if RANK in [-1, 0]:
        logger.info(f'\n{epochs - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        logger.info(
            f'\n the output of training has been output to {cfg.base_dir}, the logger in {cfg.log_file}/result.txt')

    # dataloaders
    # train_loader_cfg = cfg.data.get('')
    # build_dataloader()
    # optimizer
