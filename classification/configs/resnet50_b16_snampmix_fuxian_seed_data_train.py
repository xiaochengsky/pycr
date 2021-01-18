# resnet50 baseline

config = dict(
    # Basic cofnfig
    enable_backends_cudnn_benchmark=True,
    max_epochs=25+1,

    # 间隔多少 iter 打印一次 loss, acc 等数据
    log_periods=5,

    # 权重存储: checkpoints/tag/epoch_10.pth'
    save_dir=r"./checkpoints",

    save_weight=True,
    save_all_weight=False,

    # 当前的配置文件名字, 对应文档的操作, 比如说 aug
    tag='aug',

    # tensorX
    log_dir=r"./log",

    # 多少个 epoch 保存一次, 0 表示保存每一个 epoch
    # 最后一个 epoch 也会保存
    save_epoch_periods=0,

    # 多少个 epoch 做一次验证, 同上
    calc_epoch_periods=0,

    multi_gpus=True,
    max_num_devices=1,

    # DataLoader Config
        ## dataloader: image2batch, 继承自 torch.utils.data.DataLoader
        ## dataset: 加载 image 和 label: data/dataset/bulid.py
        ## transforms: 在线数据增强: data/transforms/opencv_transforms.py

    train_multi_scale=False,
    train_grid_size=32,
    train_pipeline=dict(
        dataloader=dict(
            batch_size=16,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
            shuffle=True,
            # collate_fn="my_collate_fn",
        ),

        dataset=dict(
            type="train_dataset",
            root_dir=r"/home/yc/opt/kaggle/Cassava-leaf-dataset",
            label_path=r"snapmix_train.txt"
            # label_path=r"train_labels_fold5.txt",
            # images_per_classes=4,
            # classes_per_minibatch=1,
        ),

        transforms=[
            # dict(type="ShiftScaleRotate", p=0.3, shift_limit=0.1, scale_limit=(-0.5, 0.2), rotate_limit=15),
            # dict(type="IAAPerspective", p=0.1, scale=(0.05, 0.15)),
            # dict(type="ChannelShuffle", p=0.1),

            dict(type="Resize", size=512), 
            dict(type="RandomCrop", size=448), 
            dict(type="RandomRotation", degrees=15), 
            dict(type="RandomCrop", size=448), 
            dict(type="RandomHorizontalFlip", p=0.5), 

            # dict(type='Rescale', output_size=512),
            # dict(type='RandomCrop', p=1., output_size=448),
            # dict(type="RandomRotation", degrees=15),
            # # dict(type='RandomCrop', p=1., output_size=448),
            # dict(type="RandomHorizontalFlip", p=0.5),
            # # dict(type="RandomVerticalFlip", p=0.5),


            # dict(type="ColorJitter", brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # dict(type="RandomErasing", p=0.2, sl=0.02, sh=0.2, rl=0.2),
            # dict(type="RandomPatch", p=0.05, pool_capacity=1000, min_sample_size=100, patch_min_area=0.01,
            #    patch_max_area=0.2, patch_min_ratio=0.2, p_rotate=0.5, p_flip_left_right=0.5),
            # dict(type="GridMask", p=0.15, drop_ratio=0.2),
            dict(type="ToTensor", ),
            dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
        ],
    ),

    val_pipeline=dict(
        dataloader=dict(
                    batch_size=32,
                    num_workers=8,
                    drop_last=False,
                    pin_memory=True,
                    shuffle=False,
                    # collate_fn="my_collate_fn",
        ),

        dataset=dict(
            type="val_dataset",
            root_dir=r"/home/yc/opt/kaggle/Cassava-leaf-dataset",
            label_path=r"snapmix_val.txt"
            # label_path=r"test_labels_fold5.txt",
            # images_per_classes=4,
            # classes_per_minibatch=1,
        ),

        transforms=[
            dict(type="Resize", size=512), 
            dict(type="CenterCrop", size=448), 

            # dict(type='Rescale', output_size=512),
            # dict(type='CenterCrop', drop_edge=32),
            # # dict(type='RescalePad', output_size=448),
            dict(type="ToTensor", ),
            dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
        ],
    ),

    # snapmix
    snapmix_pipeline=dict(
        cropsize=448,
        beta=5,
        prob=1.,
        reback="eval_cls_backbone",
    ),

    gallery_pipeline=dict(
        dataloader=dict(
            batch_size=32,
            num_workers=8,
            drop_last=False,
            pin_memory=False,
            shuffle=False,
        ),

        dataset=dict(
            type="gallery_dataset",
            root_dir=r"/opt/kaggle/Cassava-leaf-dataset",
            label_path=r"gallery_labels.txt"
        ),

        transforms=[
            dict(type="RescalePad", output_size=512),
            dict(type="ToTensor", ),
            dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
        ],
    ),

    query_pipeline=dict(
        dataloader=dict(
            batch_size=32,
            num_workers=8,
            drop_last=False,
            pin_memory=False,
            shuffle=False,
        ),

        dataset=dict(
            type="query_dataset",
            root_dir=r"",
            label_path=r""
        ),

        transforms=[
            dict(type="RescalePad", output_size=512),
            dict(type="ToTensor", ),
            dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
        ],
    ),

    # Model config
        ## backbone:            特征提取
        ## aggregation:         Pool layers. GeM, GAP or GMP and so on...
        ## heads:               classification heads,
        ## losses:              model loss
    model=dict(
        net=dict(type='ResNet50_GeM_Identity_SnapMix_CE', ),
        backbone=dict(type='resnet50', pretrained=True),
        # aggregation=dict(type="GeneralizedMeanPooling", ),
        aggregation=dict(type="AdaptiveAvgPool2d", output_size=(1, 1)),
        heads=dict(type='IdentityHead', ),
        losses=[
            dict(type='SnapMixLoss', in_feat=2048, num_classes=5, weight=1.0),
            # dict(type='CrossEntroy', in_feat=2048, num_classes=5, weight=1.0)
            # dict(type='Arcface', in_feat=2048, num_classes=5, scale=35, margin=0.30, drop_out=0.2, weights=1.0),
            # dict(type='Triplet', margin=0.60, weights=1.0),
        ]
    ),

    # Solver: 学习率调整策略, 从 torch.optim.lr_scheduler 加载
    # lr
    # lr_scheduler=dict(type="ExponentialLR", gamma=0.99998),
    # lr_scheduler=dict(type="ExponentialLR", gamma=1.),    
    # lr_scheduler=dict(type="CyclicLR", base_lr=1e-4, max_lr=4e-4, step_size_up=1000, mode='triangular2', cycle_momentum=False),
    lr_scheduler=dict(type="MultiStepLR", milestones=[20], gamma=0.1, last_epoch=-1),

    # optim
    # optimizer=dict(type="AdamW", lr=4e-4, weight_decay=1e-5),
    optimizer=dict(type="SGD", lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True),

    # warm_up
    warm_up=dict(length=0, min_lr=4e-6, max_lr=4e-4, froze_num_lyers=8),
)


if __name__ == "__main__":
    print(config)



