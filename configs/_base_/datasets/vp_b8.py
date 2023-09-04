# -*- coding: utf-8 -*-
# @Time : 2022/11/25 下午3:39
# @Author : YANG.C
# @File : vp_b8.py

# dataset settings


dataset_type = 'Vehicle'
imgsz = 512
train_pipeline = [
    dict(type='LoadImageFromFile', ),
    # dict(type='transforms_type', backend='cv2'),
    # dict(type="ShiftScaleRotate", p=0.3, shift_limit=0.1, scale_limit=(-0.5, 0.2), rotate_limit=15),
    # dict(type="IAAPerspective", p=0.1, scale=(0.05, 0.15)),
    dict(type="ChannelShuffle", p=0.1),
    # dict(type="RandomRotate90", p=0.2),
    # dict(type="RandomHorizontalFlip", p=0.5),
    # dict(type="RandomVerticalFlip", p=0.5),
    dict(type="ColorJitter", brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    dict(type="RandomErasing", p=0.2, sl=0.02, sh=0.2, rl=0.2),
    # dict(type="RandomPatch", p=0.05, pool_capacity=1000, min_sample_size=100, patch_min_area=0.01,
    #      patch_max_area=0.2, patch_min_ratio=0.2, p_rotate=0.5, p_flip_left_right=0.5),
    dict(type='RescalePad', output_size=imgsz),
    dict(type="ToTensor", ),
    dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
]

val_pipeline = [
    dict(type='LoadImageFromFile', ),
    # dict(type='transforms_type', backend='cv2'),
    dict(type="RescalePad", output_size=imgsz),
    dict(type="ToTensor", ),
    dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
]

test_pipeline = [
    dict(type='LoadImageFromFile', ),
    # dict(type='transforms_type', backend='cv2'),
    dict(type="RescalePad", output_size=imgsz),
    dict(type="ToTensor", ),
    dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,

    train=dict(
        type=dataset_type,
        data_prefix='data/VehiclePhaseTwo/vp_train_label.txt',
        pipeline=train_pipeline,
    ),

    val=dict(
        type=dataset_type,
        data_prefix='data/VehiclePhaseTwo/vp_val_label.txt',
        pipeline=val_pipeline,
    ),

    test=dict(
        type=dataset_type,
        data_prefix='data/VehiclePhaseTwo',
        pipeline=test_pipeline,
    )
)
