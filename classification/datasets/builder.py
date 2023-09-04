# -*- coding: utf-8 -*-
# @Time : 2022/12/2 上午11:07
# @Author : YANG.C
# @File : builder.py

import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from classification.utils.registry import Registry
from classification.utils.dist_utils import get_dist_info
from classification.datasets.collect.collect_fn import collect_function


DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
SAMPLES = Registry('sample')


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_pipeline(cfg):
    from .pipeline import Compose
    cfg = copy.deepcopy(cfg)
    return Compose(cfg)


def build_dataset(cfg):
    cfg = copy.deepcopy(cfg)
    dataset_type = cfg.pop('type')
    dataset = DATASETS[dataset_type](**cfg)
    return dataset


def build_sampler(cfg):
    cfg = copy.deepcopy(cfg)
    sampler_type = cfg.pop('type')
    sampler = SAMPLES[sampler_type](**cfg)
    return sampler


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     round_up=True,
                     pin_memory=True,
                     seed=None,
                     sampler_cfg=None,
                     **kwargs
                     ):
    """Build PyTorch DataLoader."""
    rank, world_size = get_dist_info()
    if sampler_cfg:
        pass
    # if dist:
    #     sampler = build_sampler(
    #         dict(
    #             type='DistributedSampler',
    #             dataset=dataset,
    #             num_replicas=world_size,
    #             rank=rank,
    #             shuffle=shuffle,
    #             round_up=round_up,
    #             seed=seed))
    # else:
    #     sampler = None

    # if sampler is not None:
    #     shuffle = False
    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed
    ) if seed is not None else None

    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     num_workers=num_workers,
    #     # collate_fn,
    #     pin_memory=pin_memory,
    #     shuffle=shuffle,
    #     worker_init_fn=init_fn,
    #     **kwargs
    # )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collect_function if kwargs['collect'] else None,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=False,
        # worker_init_fn=init_fn,
        # **kwargs
    )

    return data_loader
