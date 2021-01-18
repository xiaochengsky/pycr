
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import *


class wrapper_lr_scheduler(object):
    def __init__(self, cfg_lr_scheduler, optimizer):
        cfg_lr_scheduler_c = cfg_lr_scheduler.copy()
        lr_scheduler_type = cfg_lr_scheduler_c.pop("type")
        if hasattr(lr_scheduler, lr_scheduler_type):
            print(optimizer)
            print(lr_scheduler_type, cfg_lr_scheduler_c)
            self.lr = getattr(lr_scheduler, lr_scheduler_type)(optimizer, **cfg_lr_scheduler_c)
        else:
            raise KeyError("lr_scheduler{} not found!!!".format(lr_scheduler_type))

    def ITERATION_COMPLETED(self):
        if isinstance(self.lr, (CyclicLR, OneCycleLR, ExponentialLR)):
            self.lr.step()

    def EPOCH_COMPLETED(self):
        if isinstance(self.lr, (StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts)):
            self.lr.step()

