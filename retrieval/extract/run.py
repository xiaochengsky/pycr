import sys
import os
import datetime
import copy
import torch
from tqdm import tqdm

from ...classification.data.dataloader.build import create_dataloader
from ...classification.utils.utils import *
from .build import *
from ..configs import load_args, merge_from_arg


if __name__ == '__main__':
    init_torch_seeds(1)
    arg = vars(load_args())
    config_file = arg['config_file']

    # configs/resnet50_baseline.py => configs.resnet50_baseline
    config_file = config_file.replace("../", "").replace('.py', '').replace('/', '.')
    # print(config_file)

    # from configs.extract.r50 import config as cfg
    exec(r"from {} import config as cfg".format(config_file))
    # print(cfg['tag'], cfg['max_num_devices'])

    # 脚本输入参数替换掉字典输入
    cfg = merge_from_arg(cfg, arg)
    cfg_copy = copy.deepcopy(cfg)

    # 构建数据
    gallery_dataloader = create_dataloader(cfg['gallery_pipeline'])
    # query_dataloader = create_dataloader(cfg['query_pipeline'])
    print('gallery_dataloader: ', len(gallery_dataloader))
    # print('query_dataloader: ', len(query_dataloader))

    current_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(current_time, '%Y%m%d_')
    save_dir = os.path.join(cfg['save_dir'], time_str, cfg['tag'])
    # log_dir = os.path.join(cfg['log_dir'], "log_" + time_str + cfg['tag'])
    # cfg['save_dir'] = save_dir
    # cfg['log_dir'] = log_dir
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # if not os.path.isdir(log_dir):
    #     os.makedirs(log_dir)
    # print('Save dir: ', save_dir)
    # print('Log dir: ', log_dir)

    # 构建模型
    model = build_model(cfg, pretrain_path=arg['load_path'])

    if arg['device']:
        free_device_ids = arg['device']
    else:
        free_device_ids = get_free_device_ids()

    max_num_devices = cfg['max_num_devices']
    if len(free_device_ids) >= max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]

    master_device = free_device_ids[0]
    print('master_device: ', master_device)
    model.cuda(master_device)
    model = nn.DataParallel(model, device_ids=free_device_ids).cuda(master_device)

    if 'enable_backends_cudnn_benchmark' in cfg and cfg['enable_backends_cudnn_benchmark']:
        print("enable backends cudnn benchmark")
        torch.backends.cudnn.benchmark = True

    # 构建 extractor_type, aggregator
    extractor_type, aggregator, save_dirs = build_extractor(cfg['extract_pipeline'])

    print('save_dirs: ', save_dirs)
    save_name = config_file.split('.')[-1]
    print('save_name: ', save_name)
    if not os.path.isdir(os.path.join(save_dirs, save_name)):
        print('create dirs: {}'.format(os.path.join(save_dirs, save_name)))
        os.makedirs(os.path.join(save_dirs, save_name))

    # 抽取特征
    gallery_vectors = []
    gallery_fns = []
    gallery_targets = []

    with torch.no_grad():
        model.eval()
        for imgs, targets, img_names in tqdm(gallery_dataloader):
            vectors = model(imgs.to(master_device), extract_features_flag=True, features_type=extractor_type)

            if extractor_type == "backbone":
                vectors = aggregator(vectors)
                vectors = vectors.view((-1, vectors.shape[1],)).cpu().numpy()
            else:
                vectors = vectors.cpu().numpy()

            for i in range(len(vectors)):
                gallery_vectors.append(vectors[i])
                gallery_fns.append(img_names[i])
                gallery_targets.append(targets[i])    

    print('\n')
    print('Model[{}] save gallery features and names ======>'.format(cfg['model']['net']['type']))

    print('save {} features ======>'.format(save_name))
    np.save(os.path.join(save_dirs, save_name, 'gallery_features.npy'), gallery_vectors, allow_pickle=True)

    print('save {} img class id ======>'.format(save_name))
    np.save(os.path.join(save_dirs, save_name, "gallery_targets.npy"), gallery_targets, allow_pickle=True)

    print('save {} img names ======>'.format(save_name))
    np.save(os.path.join(save_dirs, save_name, 'gallery_names.npy'), gallery_fns, allow_pickle=True)

    print('Model[{}] is done...'.format(cfg['model']['net']['type']))
    print('\n')

    # 抽取特征
    query_vectors = []
    query_fns = []
    query_targets = []

    with torch.no_grad():
        model.eval()
        for imgs, targets, img_names in tqdm(query_dataloader):
            vectors = model(imgs.to(master_device), extract_features_flag=True, features_type=extractor_type)

            if extractor_type == "backbone":
                vectors = aggregator(vectors)
                vectors = vectors.view((-1, vectors.shape[1],)).cpu().numpy()
            else:
                vectors = vectors.cpu().numpy()

            for i in range(len(vectors)):
                query_vectors.append(vectors[i])
                query_fns.append(img_names[i])
                query_targets.append(targets[i])

    print('\n')
    print('Model[{}] save query features and names ======>'.format(cfg['model']['net']['type']))
    
    print('save {} features ======>'.format(save_name))
    np.save(os.path.join(save_dirs, save_name, 'query_features.npy'), query_vectors, allow_pickle=True)

    print('save {} img class id ======>'.format(save_name))
    np.save(os.path.join(save_dirs, save_name, "query_targets.npy"), query_targets, allow_pickle=True)

    print('save {} img names ======>'.format(save_name))
    np.save(os.path.join(save_dirs, save_name, 'query_names.npy'), query_fns, allow_pickle=True)

    print('Model[{}] is done...'.format(cfg['model']['net']['type']))
    print('\n')

