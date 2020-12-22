import sys
import os
import datetime
import copy

sys.path.append('..')

from ..data.dataloader.build import create_dataloader
from ..model.build import build_model
from ..solver.optimizer import create_optimizer
from ..solver.lr_scheduler import wrapper_lr_scheduler
from ..utils.utils import *
from ..engine.trainer import *
from ..configs import load_args, merge_from_arg

# from data.dataloader import create_dataloader


if __name__ == '__main__':

    init_torch_seeds(1)

    arg = vars(load_args())
    config_file = arg['config_file']

    # configs/resnet50_baseline.py => configs.resnet50_baseline
    config_file = config_file.replace("../", "").replace('.py', '').replace('/', '.')
    # print(config_file)

    # from configs.resnet50_baseline import config as cfg
    exec(r"from {} import config as cfg".format(config_file))
    # print(cfg['tag'], cfg['max_num_devices'])

    # 脚本输入参数替换掉字典输入
    cfg = merge_from_arg(cfg, arg)
    cfg_copy = copy.deepcopy(cfg)

    train_dataloader = create_dataloader(cfg['train_pipeline'])
    val_dataloader = create_dataloader(cfg['val_pipeline'])

    print('train_dataloader: ', len(train_dataloader))
    print('val_dataloader: ', len(val_dataloader))

    current_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(current_time, '%Y%m%d_')
    save_dir = os.path.join(cfg['save_dir'], time_str, cfg['tag'])
    log_dir = os.path.join(cfg['log_dir'], "log_" + time_str + cfg['tag'])
    cfg['save_dir'] = save_dir
    cfg['log_dir'] = log_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print('Save dir: ', save_dir)
    print('Log dir: ', log_dir)

    model = build_model(cfg, pretrain_path=arg['load_path'])
    optimizer = create_optimizer(cfg['optimizer'], model)
    lr_scheduler = wrapper_lr_scheduler(cfg['lr_scheduler'], optimizer)

    if arg['device']:
        free_device_ids = arg['device']
    else:
        free_device_ids = get_free_device_ids()

    max_num_devices = cfg['max_num_devices']
    if len(free_device_ids) >= max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]

    master_device = free_device_ids[0]
    model.cuda(master_device)
    model = nn.DataParallel(model, device_ids=free_device_ids).cuda(master_device)

    cfg_copy['save_dir'] = save_dir  # 更新存储目录
    cfg_copy['log_dir'] = log_dir  # 更新存储目录

    do_train(cfg_copy, model=model, train_loader=train_dataloader, val_loader=val_dataloader, optimizer=optimizer,
             scheduler=lr_scheduler, device=free_device_ids)
