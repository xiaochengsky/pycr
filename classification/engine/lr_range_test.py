import logging
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Timer, TerminateOnNan, ModelCheckpoint
from ignite.metrics import Loss, RunningAverage, Accuracy
import torch
import os
from tqdm import tqdm
import numpy as np
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torch.utils.tensorboard import SummaryWriter
from ..model.net import freeze_layers, fix_bn
from torch.cuda.amp import autocast as autocast
from ..utils.utils import *
import torch.nn as nn

import logging

global ITER, ALL_ITER, ALL_ACC
ITER = 0
ALL_ITER = 0
ALL_ACC = []


def do_lr_range_test(cfg, model, train_loader, val_loader, optimizer, scheduler, device):
    scaler = torch.cuda.amp.GradScaler()
    log_dir = cfg['log_dir']
    max_epochs = cfg['max_epochs']

    writer = SummaryWriter(log_dir=log_dir)

    # logging.basicConfig(level=logging.DEBUG)
    # logger = logging.getLogger("Cassava")
    # logger.info("Start training ======>")
    all_iter = 0
    lr_mult = (1 / 1e-5) ** (1 / 100)
    for x, y, img_names in train_loader:
        all_iter += 1
        x.to(device)
        y.to(device)
        model.to(device)
        model.train()
        optimizer.zero_grad()
        with autocast():
            total_loss = model(x, y)
        total_loss = total_loss.mean()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # get lr
        curr_lr = optimizer.param_groups[0]['lr']

        num_correct, num_example = 0, 0
        with torch.no_grad():
            model.eval()
            for image, target, image_name in tqdm(val_loader):
                image, target = image.to(device), target.to(device)
                pred_logit = model(image, target)
                indics = torch.max(pred_logit, dim=1)[1]
                correct = torch.eq(indics, target).view(-1)
                num_correct += torch.sum(correct).item()
                num_example += correct.shape[0]
        acc = num_correct / num_example

        writer.add_scalar('learning rate', curr_lr, all_iter)
        writer.add_scalar("total loss", total_loss.cpu().data.numpy(), all_iter)
        writer.add_scalar("current acc", acc, all_iter)
        print('Iter: {} | Lr: {} | Loss:{} | Acc: {}'.format(all_iter, curr_lr, total_loss.cpu().data.numpy(), acc))
        # change lr
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

        if curr_lr >= 0.1:
            break
