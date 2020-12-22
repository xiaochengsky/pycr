import torch
import torch.nn as nn
import os
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from ..model import backbone as backbones
from ..model import aggregation as aggregations
from ..model import head as heads
from ..model import losses as losses
from ..model import layers as layer
from ..utils.utils import *


# build bockbones
def build_backbone(cfg_backbone):
    backbone_type = cfg_backbone.pop('type')
    if hasattr(backbones, backbone_type):
        backbone = getattr(backbones, backbone_type)(**cfg_backbone)
        return backbone
    else:
        raise KeyError("backbond_type{} is not found!!!".format(backbone_type))


# build aggregations
def build_aggregation(cfg_aggregation):
    aggregation_type = cfg_aggregation.pop('type')

    # GAP / GMP and so on...
    if hasattr(nn, aggregation_type):
        aggregation = getattr(nn, aggregation_type)(**cfg_aggregation)
        return aggregation
    elif hasattr(aggregations, aggregation_type):
        aggregation = getattr(aggregations, aggregation_type)(**cfg_aggregation)
        return aggregation
    else:
        raise KeyError("aggregation_type{} is not found!!!".format(aggregation_type))


# build heads
def build_head(cfg_heads):
    head_type = cfg_heads.pop("type")
    print('head_type: ', head_type)
    if hasattr(nn, head_type):
        print('nnnnnnnn')
        head = getattr(nn, head_type)(**cfg_heads)
        return head
    elif hasattr(heads, head_type):
        head = getattr(heads, head_type)(**cfg_heads)
        print('customers')
        return head
    else:
        print('* ' * 20)
        return KeyError("head_type{} is not found!!!".format(head_type))


# fix bn
def fix_bn(model):
    print("------------fix bn start---------------")
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


# freeze layers
def freeze_layers(model, num_layers=0):
    length_backbone = len(list(model.backbone.named_children()))
    print('freeze layers of backbone: ', length_backbone)
    if length_backbone == 1:
        pass
    elif length_backbone <= 2:
        pass

    # resnet
    else:
        for i, (name, child) in enumerate(model.backbone.named_children()):
            if i < num_layers:
                child.eval()
                for param in child.parameters():
                    param.requires_grad = False
                print("freeze: ", name)
            else:
                child.train()
                for param in child.parameters():
                    param.requires_grad = True
                print("unfreeze: ", name)

