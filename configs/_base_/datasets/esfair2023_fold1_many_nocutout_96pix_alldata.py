# -*- coding: utf-8 -*-
# @Time: 2023/5/18 下午5:48
# @Author: YANG.C
# @File: esfair2023.py

dataset_type = 'ESFair'
imgsz = 96

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # # dict(type='transforms_type', backend='cv2'),
    # dict(type="Transpose", p=0.2), 
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="RandomVerticalFlip", p=0.5),
    # dict(type="RandomBrightnessContrast", p=0.2), 
    # dict(type="OneOfBlur", p=0.1), 
    # dict(type="OneOfDistortion", p=0.1),
    # dict(type="CLAHE", p=0.2),
    # dict(type="HueSaturationValue", p=0.2, hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
    # dict(type="ShiftScaleRotate", p=0.1, shift_limit=0.1, scale_limit=(-0.1, 0.1), rotate_limit=15),
    dict(type="IAAPerspective", p=0.1, scale=(0.05, 0.15)),
    # dict(type="RandomErasing", p=0.2, sl=0.02, sh=0.2, rl=0.2),
    # # dict(type="RandomPatch", p=0.05, pool_capacity=1000, min_sample_size=100, patch_min_area=0.01,
    # #      patch_max_area=0.2, patch_min_ratio=0.2, p_rotate=0.5, p_flip_left_right=0.5),
    dict(type="RescalePad", output_size=imgsz),
    # dict(type="DrawHair", p=0.2), 
    # dict(type="Cutout", max_h_size=int(imgsz * 0.2), max_w_size=int(imgsz * 0.2), num_holes=1, p=0.2), 
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
    samples_per_gpu=64,
    workers_per_gpu=12,

    train=dict(
        type=dataset_type,
        data_prefix='data/esfair/fold_1/all_esfair_train_fold1.txt',
        pipeline=train_pipeline,
    ),

    val=dict(
        type=dataset_type,
        data_prefix='data/esfair/fold_1/esfair_val_fold1.txt',
        pipeline=val_pipeline,
    ),

    test=dict(
        type=dataset_type,
        data_prefix='data/esfair/esfair_val_label.txt',
        pipeline=test_pipeline,
    )
)
