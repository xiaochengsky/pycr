# -*- coding: utf-8 -*-
# @Time : 2022/11/28 下午4:08
# @Author : YANG.C
# @File : train.py

import argparse
import logging
import sys
import os

sys.path.append('../pycr_tc')

from classification.utils.config import Config, DictAction
from utils.log import logger, format
from classification.api.train import train_model
from classification.utils.dist_utils import init_seeds

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


#
# logger.debug(f'LOCAL_RANK: {LOCAL_RANK}, RANK: {RANK}, WORLD_SIZE: {WORLD_SIZE}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--model', help='build model'
    )
    return parser.parse_args()


def log_config(cfg):
    # logger.debug(cfg)
    # mkdir dirs
    logs = './logs'
    tag = cfg.tag
    base_dir = f'{logs}/{cfg.model.backbone.type}_{tag}'
    tensorboard_dir = f'{base_dir}/{cfg.tensorboard_dir}'
    checkpoint_dir = f'{base_dir}/{cfg.save_dir}'
    cfg.__setitem__('base_dir', base_dir)
    cfg.__setitem__('tensorboard_dir', tensorboard_dir)
    cfg.__setitem__('checkpoint_dir', checkpoint_dir)

    log_dir = f'{base_dir}/{cfg.log_file}'
    if RANK in [-1, 0]:
        for dir in [base_dir, tensorboard_dir, checkpoint_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # redirect to local file
        fh = logging.FileHandler(log_dir)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(format)
        logger.addHandler(fh)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    log_config(cfg)

    # train model
    train_model(cfg)


if __name__ == '__main__':
    main()
