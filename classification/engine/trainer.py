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
from model.net import freeze_layers, fix_bn
from torch.cuda.amp import autocast as autocast
from ..utils.utils import *
import torch.nn as nn

import logging

global ITER, ALL_ITER, ALL_ACC
ITER = 0
ALL_ITER = 0
ALL_ACC = []


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, device):
    scaler = torch.cuda.amp.GradScaler()
    log_dir = cfg['log_dir']
    max_epochs = cfg['max_epochs']

    writer = SummaryWriter(log_dir=log_dir)

    # logging.basicConfig(level=logging.DEBUG)
    # logger = logging.getLogger("Cassava")
    # logger.info("Start training ======>")

    def _prepare_batch(batch, device=None, non_blocking=False):
        x, y, img_names = batch
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

        def _update(engine, batch):
            model.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

            # amp
            with autocast():
                total_loss = model(x, y)

            total_loss = total_loss.mean()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            writer.add_scalar("total loss", total_loss.cpu().data.numpy())
            scaler.update()

            # 返回 loss.item()
            return output_transform(x, y, None, total_loss)

        return Engine(_update)

    # 选择主卡计算 loss
    master_device = device[0]

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
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                break
            # logger.info("Epoch[{}], Iteration[{}/{}], Loss: {:.3f}, Base Lr: {:.2e}"
            #             .format(engine.state.epoch, ITER, len(train_loader),
            #                     engine.state.metrics['avg_loss'], curr_lr))
            print("Epoch[{}], Iteration[{}/{}], Loss: {:.3f}, Base Lr: {:.2e}"
                  .format(engine.state.epoch, ITER, len(train_loader),
                          engine.state.metrics['avg_loss'], curr_lr))
            writer.add_scalar('loss', engine.state.metrics['avg_loss'], ALL_ITER)

    # lr_scheduler warm up
    @trainer.on(Events.ITERATION_COMPLETED)
    def lr_scheduler_iteration(engine):
        global ALL_ITER

        scheduler.ITERATION_COMPLETED()
        length = cfg['warm_up']['length']
        min_lr = cfg['warm_up']['min_lr']
        max_lr = cfg['warm_up']['max_lr']

        # warm up
        if ALL_ITER < length:
            lr = (max_lr - min_lr) / length * ALL_ITER
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # if ALL_ITER == length:
        #     pass

    # EPOCH_COMPLETED : triggered when the epoch is ended
    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_scheduler_epoch(engine):
        scheduler.EPOCH_COMPLETED()

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

            # Acc
            writer.add_scalar("Acc", acc, epoch)
            # cnf
            class_names = ['CBB(0)', 'CBSD(1)', 'CGM(2)', 'CMD(3)', 'Healthy(4)']
            writer.add_figure('confusion matrix_' + str(epoch),
                              figure=plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                                                           title='show confusion matrix'), global_step=1)

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
            sava_model_path = os.path.join(save_dir, tag, "epoch_" + str(epoch) + ".pth")
            save_model_weight_path = os.path.join(save_dir, tag, "weight_epoch_" + str(epoch) + ".pth")
            # https://www.codeleading.com/article/12702208128/
            if cfg['multi_gpus']:
                save_path = {'model': model.module.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'epoch': epoch,
                             'tag': tag,
                             'acc': ALL_ACC[epoch - 1],
                             'cfg': cfg}
                save_weight_path = {'model': model.module.state_dict(),
                                    'epoch': epoch,
                                    'tag': tag,
                                    'acc': ALL_ACC[epoch-1],
                                    'cfg': cfg}
            else:
                # print('len(ALL_ACC): ', len(ALL_ACC))
                # print('epoch: ', epoch)
                save_path = {'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'epoch': epoch,
                             'tag': tag,
                             'acc': ALL_ACC[epoch - 1],
                             'cfg': cfg}
                save_weight_path = {'model': model.state_dict(),
                                    'epoch': epoch,
                                    'tag': tag,
                                    'acc': ALL_ACC[epoch-1],
                                    'cfg': cfg}
            torch.save(save_weight_path, save_model_weight_path)
            torch.save(save_path, sava_model_path)

    max_epochs = cfg['max_epochs']
    trainer.run(train_loader, max_epochs=max_epochs)
