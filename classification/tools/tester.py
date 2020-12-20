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

# from data.dataloader import create_dataloader
from ..configs import load_args, merge_from_arg

if __name__ == '__main__':
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

    model = build_model(cfg, pretrain_path=arg['load_path'])
    # optimizer = create_optimizer(cfg['optimizer'], model)
    # lr_scheduler = wrapper_lr_scheduler(cfg['lr_scheduler'], optimizer)

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

    if 'enable_backends_cudnn_benchmark' in cfg and cfg['enable_backends_cudnn_benchmark']:
        print("enable backends cudnn benchmark")
        torch.backends.cudnn.benchmark = True

    # calc_acc
    num_corrects = 0
    num_examples = 0
    model.eval()
    with torch.no_grad():
        for image, target, image_name in tqdm(val_dataloader):
            image, target = image.to(master_device), target.to(master_device)
            pred_logit = model(image, target)
            indics = torch.max(pred_logit, dim=1)[1]
            correct = torch.eq(indics, target).view(-1)
            num_corrects += torch.sum(correct).item()
            num_examples += correct.shape[0]

        print('Acc: ', num_corrects / num_examples)
