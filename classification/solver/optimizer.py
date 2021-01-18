from torch import optim


def create_optimizer(cfg_optimizer, model):
    cfg_optimizer_c = cfg_optimizer.copy()
    optimizer_type = cfg_optimizer_c.pop("type")
    if hasattr(optim, optimizer_type):
        # 获取模型所有参数
        params = model.parameters()
        print(cfg_optimizer_c)

        param_type = []
        param_groups = []
        # 为不同的层设置不同的 lr
        for name, param in model.named_parameters():
            if 'backbone' in name and 'backbone' not in param_type:
                param_type.append('backbone')
                param_groups.append({'params': model.backbone.parameters(), 'lr': cfg_optimizer_c['lr'] * 0.1})
            if 'layer' in name and 'layer' not in param_type:
                param_type.append('layer')
                param_groups.append({'params': model.layer.parameters(), 'lr': cfg_optimizer_c['lr']})
            if 'aggregation' in name and 'aggregation' not in param_type:
                param_type.append('aggregation')
                param_groups.append({'params': model.aggregation.parameters(), 'lr': cfg_optimizer_c['lr']})
            if 'celoss' in name and 'celoss' not in param_type:
                param_type.append('celoss')
                param_groups.append({'params': model.celoss.parameters(), 'lr': cfg_optimizer_c['lr']})

        print(param_type)
        print(param_groups)
        # param_groups = [
        #    {'params': model.backbone.parameters(), 'lr': cfg_optimizer_c['lr'] * 0.1},
        #    {'params': model.celoss.parameters(), 'lr': cfg_optimizer_c['lr']},
        # ]
        optimizer = getattr(optim, optimizer_type)(param_groups, **cfg_optimizer_c)
        lr = []
        i = 0
        for param_group in optimizer.param_groups:
            if i == 0:
                print('backbone')
            else:
                print('else')
            lr += [param_group['lr']]
            i += 1
        print(lr)

        return optimizer
    else:
        raise KeyError("optimizer{} is not found!!!".format(optimizer_type))
