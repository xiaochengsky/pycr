
from torch import optim


def create_optimizer(cfg_optimizer, model):
    cfg_optimizer_c = cfg_optimizer.copy()
    optimizer_type = cfg_optimizer_c.pop("type")
    if hasattr(optim, optimizer_type):
        # 获取模型所有参数
        params = model.parameters()
        optimizer = getattr(optim, optimizer_type)(params, **cfg_optimizer_c)
        return optimizer
    else:
        raise KeyError("optimizer{} is not found!!!".format(optimizer_type))


