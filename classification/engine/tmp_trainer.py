import logging
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Timer, TerminateOnNan, ModelCheckpoint
from ignite.metrics import Loss, RunningAverage, Accuracy
import torch
import os
from tqdm import tqdm
import numpy as np
import random
from torch.optim.lr_scheduler import *

from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from ..model.components import freeze_layers, fix_bn
from ..utils.utils import *
from ..utils.snapmix import *


np.set_printoptions(suppress=True)

global ITER, ALL_ITER, ALL_ACC
ITER = 0
ALL_ITER = 0
ALL_ACC = []
import pdb
# pdb.set_trace()

def do_train(cfg, model, ema_model, train_loader, val_loader, optimizer, scheduler, device):
    # scaler = torch.cuda.amp.GradScaler()
    log_dir = cfg['log_dir']
    max_epochs = cfg['max_epochs']

    writer = SummaryWriter(log_dir=log_dir)

    # 选择主卡计算 loss
    master_device = device[0]

    # EMA init
    if cfg['multi_gpus']:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    ema_model.load_state_dict(model_state_dict)
    ema = ModelEMA(ema_model)

    def _prepare_batch(batch, device=None, non_blocking=False):
        x, y, img_names = batch
        # Multi-Scale
        if cfg['train_multi_scale']:
            if 'train_grid_size' in cfg:
                gs = cfg['train_grid_size']
            else:
                gs = 32
            sz = random.randrange(x.shape[2] * 0.75, x.shape[2] * 1.25 + gs) // gs * gs  # size
            sf = sz / max(x.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(s * sf / gs) * gs for s in x.shape[2:]]
                x = F.interpolate(x, size=ns, mode='bilinear', align_corners=False)
        return (convert_tensor(x, device=device, non_blocking=non_blocking),
                convert_tensor(y, device=device, non_blocking=non_blocking))

    def create_supervised_dp_trainer(model, optimizer, device=None, non_blocking=False,
                                     prepare_batch=_prepare_batch,
                                     output_transform=lambda x, y, y_pred, loss: loss.item()):

        """
        Factory function for creating a trainer for supervised models.

        Args:
            model (`torch.nn.Module`): the model to train.
            optimizer (`torch.optim.Optimizer`): the optimizer to use.
            loss_fn (torch.nn loss function): the loss function to use.
            device (str, optional): device type specification (default: None).
                Applies to batches after starting the engine. Model *will not* be moved.
                Device can be CPU, GPU or TPU.
            non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
                with respect to the host. For other cases, this argument has no effect.
            prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
                tuple of tensors `(batch_x, batch_y)`.
            output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
                to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
            deterministic (bool, optional): if True, returns deterministic engine of type
                :class:`~ignite.engine.deterministic.DeterministicEngine`, otherwise :class:`~ignite.engine.engine.Engine`
                (default: False).

        Note:
            `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
            of the processed batch by default.

        Returns:
            Engine: a trainer engine with supervised update function.
        """
        if device:
            model.to(device)
            ema_model.to(device)

        def _update(engine, batch):
            model.train()
            # optimizer.zero_grad()
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

            # amp
            # with autocast():
            if "snapmix_pipeline" in cfg.keys():
                cfg_snapmix = cfg["snapmix_pipeline"]
                x, ya, yb, lam_a, lam_b = snapmix(x, y, cfg_snapmix, model)
                total_loss = model(x, ya=ya, yb=yb, lam_a=lam_a, lam_b=lam_b)
            else:
                total_loss = model(x, y)
                total_loss = total_loss.mean()
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # scaler.scale(total_loss).backward()
            # scaler.step(optimizer)
            writer.add_scalar("total loss", total_loss.cpu().data.numpy())
            # scaler.update()
            ema.update(model)

            # 返回 loss.item()
            return output_transform(x, y, None, total_loss)

        return Engine(_update)

    # 创建训练 engine
    trainer = create_supervised_dp_trainer(model, optimizer, device=master_device)

    # 注册回调事件: 迭代完毕检查 loss
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    # ITERATION_COMPLETED : triggered when the iteration is ended
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER, ALL_ITER
        ITER += 1
        ALL_ITER += 1

        log_period = cfg['log_periods']
        if ITER % log_period == 0:
            i = 0
            for param_group in optimizer.param_groups:
                if i == 0:
                    ft_lr = param_group['lr']
                else:
                    rf_lr = param_group['lr']
                i += 1
            print("Epoch[{}], Iteration[{}/{}], Loss: {:.3f}, Ft Lr: {:.2e}, Rf Lr: {:.2e}"
                  .format(engine.state.epoch, ITER, len(train_loader),
                          engine.state.metrics['avg_loss'], ft_lr, rf_lr))
            writer.add_scalar('loss', engine.state.metrics['avg_loss'], ALL_ITER)

    # EPOCH_COMPLETED : triggered when the epoch is ended
    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_scheduler_epoch(engine):
        scheduler.EPOCH_COMPLETED()

    # lr_scheduler warm up
    @trainer.on(Events.ITERATION_COMPLETED)
    def lr_scheduler_iteration(engine):
        global ALL_ITER

        scheduler.ITERATION_COMPLETED()
        # print('scheduler: ', scheduler)
        # print('scheduler.lr: ', scheduler.lr)
        length = cfg['warm_up']['length']
        min_lr = cfg['warm_up']['min_lr']
        max_lr = cfg['warm_up']['max_lr']

        # warm up
        if ALL_ITER < length:
            lr = (max_lr - min_lr) / length * ALL_ITER
            i = 0
            for param_group in optimizer.param_groups:
                if i == 0:
                    param_group['lr'] = lr * 0.1
                else:                   
                    param_group['lr'] = lr
                i += 1
        else:
            if isinstance(scheduler.lr, (CyclicLR, OneCycleLR, ExponentialLR)):
                i = 0
                for param_group in optimizer.param_groups:
                    if i > 0:
                        curr_lr = param_group['lr']
                    i += 1
                i = 0
                for param_group in optimizer.param_groups:
                    if i == 0:
                        param_group['lr'] = curr_lr * 0.1
                    else:
                        break
                    i += 1

        # if ALL_ITER == length:
        #     pass

    @trainer.on(Events.EPOCH_COMPLETED)
    def clac_acc(engine):
        global ALL_ACC
        global ITER
        ITER = 0
        calc_acc_epoch = cfg['calc_epoch_periods']
        epoch = engine.state.epoch
        if epoch % (calc_acc_epoch + 1) == 0:
            model.eval()
            num_correct = 0
            num_example = 0
            torch.cuda.empty_cache()

            # cnf_matrix
            cnf_matrix = np.zeros((5, 5))

            with torch.no_grad():
                for image, target, image_name in tqdm(val_loader):
                    image, target = image.to(master_device), target.to(master_device)
                    pred_logit = model(image, target)
                    indics = torch.max(pred_logit, dim=1)[1]
                    correct = torch.eq(indics, target).view(-1)
                    num_correct += torch.sum(correct).item()
                    num_example += correct.shape[0]

                    # calc cnf_matrix
                    p_t = get_cnf_matrix(pred_logit, target)
                    cnf_matrix += p_t

            acc = num_correct / num_example
            ALL_ACC.append(acc)
            print("Acc: ", acc)
            print('Epoch: ', epoch)
            print('cnf_matrix: ')
            print(cnf_matrix)

            # Acc
            writer.add_scalar("Acc", acc, epoch)
            # cnf
            class_names = ['CBB(0)', 'CBSD(1)', 'CGM(2)', 'CMD(3)', 'Healthy(4)']
            # writer.add_figure('confusion matrix_' + str(epoch),
            #                   figure=plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
            #                                                title='show confusion matrix'), global_step=1)

            torch.cuda.empty_cache()
            model.train()

        if 'mode' in cfg and cfg['mode'] == "Finetuning":
            if cfg['multi_gpu']:
                fix_bn(model.module)
            else:
                fix_bn(model)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoints(engine):
        global ALL_ACC

        save_dir = cfg['save_dir']
        tag = cfg['tag']
        save_epoch = cfg['save_epoch_periods']

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        epoch = engine.state.epoch
        # print('epoch: ', epoch)
        if epoch % (save_epoch + 1) == 0 or epoch >= max_epochs - 2:
            save_model_dir = os.path.join(save_dir, tag)
            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)

            save_all_weight_path = os.path.join(save_dir, tag, "epoch_" + str(epoch) + ".pth")
            save_ema_all_weight_path = os.path.join(save_dir, tag, "ema_epoch_" + str(epoch) + ".pth")

            save_weight_path = os.path.join(save_dir, tag, "weight_epoch_" + str(epoch) + ".pth")
            save_ema_weight_path = os.path.join(save_dir, tag, "ema_weight_epoch_" + str(epoch) + ".pth")

            # https://www.codeleading.com/article/12702208128/
            if cfg['multi_gpus']:
                save_all_weight = {'model': model.module.state_dict(),
                                   'optimizer': optimizer.state_dict(),
                                   'epoch': epoch,
                                   'tag': tag,
                                   'acc': ALL_ACC[epoch - 1],
                                   'cfg': cfg}
                save_ema_all_weight = {'model': ema.ema.state_dict(),
                                       'optimizer': optimizer.state_dict(),
                                       'epoch': epoch,
                                       'tag': tag,
                                       'acc': ALL_ACC[epoch - 1],
                                       'cfg': cfg}

                save_weight = {'model': model.module.state_dict(),
                               'epoch': epoch,
                               'tag': tag,
                               'acc': ALL_ACC[epoch - 1],
                               'cfg': cfg}
                save_ema_weight = {'model': ema.ema.state_dict(),
                                   'epoch': epoch,
                                   'tag': tag,
                                   'acc': ALL_ACC[epoch - 1],
                                   'cfg': cfg}
            else:
                # print('len(ALL_ACC): ', len(ALL_ACC))
                # print('epoch: ', epoch)
                save_all_weight = {'model': model.state_dict(),
                                   'optimizer': optimizer.state_dict(),
                                   'epoch': epoch,
                                   'tag': tag,
                                   'acc': ALL_ACC[epoch - 1],
                                   'cfg': cfg}
                save_ema_all_weight = {'model': ema.ema.state_dict(),
                                       'optimizer': optimizer.state_dict(),
                                       'epoch': epoch,
                                       'tag': tag,
                                       'acc': ALL_ACC[epoch - 1],
                                       'cfg': cfg}

                save_weight = {'model': model.state_dict(),
                               'epoch': epoch,
                               'tag': tag,
                               'acc': ALL_ACC[epoch - 1],
                               'cfg': cfg}
                save_ema_weight = {'model': ema.ema.state_dict(),
                                   'epoch': epoch,
                                   'tag': tag,
                                   'acc': ALL_ACC[epoch - 1],
                                   'cfg': cfg}
            if cfg['save_weight']:
                torch.save(save_weight, save_weight_path)
                torch.save(save_ema_weight, save_ema_weight_path)
            if cfg['save_all_weight']:
                torch.save(save_all_weight, save_all_weight_path)
                torch.save(save_ema_all_weight, save_ema_all_weight_path)

    max_epochs = cfg['max_epochs']
    trainer.run(train_loader, max_epochs=max_epochs)
