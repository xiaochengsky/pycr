import copy
import torch
from ..model import net as net


def build_model(cfg, pretrain_path=""):
    cfg_c = copy.deepcopy(cfg)
    if 'net' in cfg["model"].keys():
        net_cfg = cfg['model']['net']
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


