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
# import model.backbone as backbones
# import model.aggregation as aggregations
# import model.head as heads
# import model.losses as losses
# from model.layers import *
# from utils.utils import weights_init_kaiming, weights_init_classifier


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


###########################################################
#################### 此处之后写各个网络实现 ##################
###########################################################

class ResNet50_GeM_Identity_CE(nn.Module):
    def __init__(self, cfg):
        super(ResNet50_GeM_Identity_CE, self).__init__()
        self.cfg = cfg
        cfg_model = self.cfg['model']
        cfg_model = deepcopy(cfg_model)

        # get configurations
        cfg_backbone = cfg_model['backbone']
        cfg_aggregation = cfg_model['aggregation']
        cfg_head = cfg_model['heads']
        cfg_losses = cfg_model['losses']

        # log
        log_dir = cfg['log_dir']
        self.write = SummaryWriter(log_dir=log_dir)
        self.step = 0

        # net
        self.backbone = build_backbone(cfg_backbone)
        self.aggregation = build_aggregation(cfg_aggregation)
        self.head = build_head(cfg_head)

        # loss
        cfg_ce = cfg_losses[0]
        loss_type = cfg_ce.pop("type")
        self.celoss = getattr(losses, loss_type)(**cfg_ce)

    def forward(self, inputs, targets=None, extract_features_flag=False, features_type="after"):
        assert features_type in ("b_features", "before", "after", "both")

        b_features = self.backbone(inputs)

        if extract_features_flag and (features_type == 'b_features'):
            # [N, C, H, W]
            return b_features

        features = self.aggregation(b_features)

        # 从 Head 之前抽取特征
        if extract_features_flag and (features_type == 'before'):
            # [N, C, 1, 1] -> [N, C]
            return features[..., 0, 0]

        head_features = self.head(features)

        # 从 Head 之后抽取
        if extract_features_flag and (features_type == 'after'):
            # [N, C]
            return head_features

        # both
        if extract_features_flag and (features_type == 'both'):
            return features[..., 0, 0], head_features

        if self.training:
            # 获得 loss
            ce_value = self.celoss(head_features, targets.long())
            # print('ce loss: ', ce_value.cpu().data.numpy())
            total_loss = torch.unsqueeze(ce_value, 0)
            # print('total_loss: ', total_loss)
            return total_loss
        else:
            # 获得 fc 的输出
            # loss_dict = {}
            # pred_logit = self.celoss(head_features, targets.long())
            # loss_dict['logit'] = pred_logit
            # return loss_dict
            pred_logit = self.celoss(head_features, targets.long())
            return pred_logit
