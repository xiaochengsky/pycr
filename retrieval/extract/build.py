# 定义 extract 类别
## 完成 model 的定义
## 完成 extractor 的定义(抽取哪一层的特征)
## 完成 aggregator 的定义(怎么聚合特征)


import copy
import torch

from .aggregator import poolings as poolings
from ...classification.model import nets as net


def build_model(cfg, pretrain_path=""):
    cfg_c = copy.deepcopy(cfg)
    if 'net' in cfg_c["model"].keys():
        net_cfg = cfg_c['model']['net']
        net_type = net_cfg.pop('type')
        model = getattr(net, net_type)(cfg_c)
    else:
        raise KeyError("net{} is not found!!!".format(cfg))

    if pretrain_path:
        model_state_dict = model.state_dict()
        state_dict = torch.load(pretrain_path, map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        for key in state_dict.keys():
            if key in model_state_dict.keys() and state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key] = state_dict[key]

        # assert
        assert len(state_dict.keys()) == len(model_state_dict.keys())
        model.load_state_dict(model_state_dict)

    return model


def build_extractor(cfg):
    cfg_c = cfg.copy()
    extractor = cfg_c.pop('extractor')
    extractor_type = extractor['extractor_type']
    assert extractor_type in ("backbone", "before", "after", "both")

    save_dirs = cfg_c.pop('save_dirs')

    aggregator = cfg_c.pop('aggregator')
    aggregator_type = aggregator['aggregator_type']
    print("aggregator_type: ", aggregator_type)
    if hasattr(poolings, aggregator_type):
        print('exist the aggregator_type')
        aggregator = getattr(poolings, aggregator_type)
    else:
        return KeyError("aggregator_type{} is not found!!!".format(aggregator_type))

    return extractor_type, aggregator, save_dirs['dir']

