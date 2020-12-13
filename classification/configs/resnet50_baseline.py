# resnet50 baseline

config = dict(
    # Basic cofnfig
    enable_backends_cudnn_benckmark=True,
    max_epochs=100 + 1,

    # 间隔多少 iter 打印一次 loss, acc 等数据
    log_periods=5,

    # 权重存储: checkpoints/tag/epoch_10.pth'
    save_dir=r"./checkpoints",
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
    max_num_devices=4,

    # DataLoader Config
        ## dataloader: image2batch, 继承自 torch.utils.data.DataLoader
        ## dataset: 加载 image 和 label: data/dataset/bulid.py
        ## transforms: 在线数据增强: data/transforms/opencv_transforms.py
    train_pipeline=dict(
        dataloader=dict(
            batch_size=8,
            num_workers=8,
            drop_last=False,
            pin_memory=False,
            # collate_fn="my_collate_fn",
        ),

        dataset=dict(
            type="train_dataset",
            root_dir=r"/home/cheng.yang/PyCharmProjects/Cassava/test_set",
            # images_per_classes=4,
            # classes_per_minibatch=1,
        ),

        transforms=[
            # dict(type="ShiftScaleRotate", p=0.3, shift_limit=0.1, scale_limit=(-0.5, 0.2), rotate_limit=15),
            # dict(type="IAAPerspective", p=0.1, scale=(0.05, 0.15)),
            # dict(type="ChannelShuffle", p=0.1),
            # dict(type="RandomRotate90", p=0.2),
            # dict(type="RandomHorizontalFlip", p=0.5),
            # dict(type="RandomVerticalFlip", p=0.5),
            # dict(type="ColorJitter", brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # dict(type="RandomErasing", p=0.2, sl=0.02, sh=0.2, rl=0.2),
            # dict(type="RandomPatch", p=0.05, pool_capacity=1000, min_sample_size=100, patch_min_area=0.01,
            #      patch_max_area=0.2, patch_min_ratio=0.2, p_rotate=0.5, p_flip_left_right=0.5),
            dict(type="ToTensor", ),
            dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
        ],
    ),

    val_pipeline=dict(
        dataloader=dict(
                    batch_size=8,
                    num_workers=8,
                    drop_last=False,
                    pin_memory=False,
                    # collate_fn="my_collate_fn",
        ),

        dataset=dict(
            type="val_dataset",
            root_dir=r"/home/cheng.yang/PyCharmProjects/Cassava/test_set",
            # images_per_classes=4,
            # classes_per_minibatch=1,
        ),

        transforms=[
            # dict(type="RescalePad",output_size=320),
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
        net=dict(type='ResNet50_GeM_Identity_CE', ),
        backbone=dict(type='resnet50', pretrained=False),
        aggregation=dict(type="GeneralizedMeanPooling", ),
        heads=dict(type='IdentityHead', ),
        losses=[
            dict(type='CrossEntroy', in_feat=2048, num_classes=5, weight=1.0)
            # dict(type='Arcface', in_feat=2048, num_classes=5, scale=35, margin=0.30, drop_out=0.2, weights=1.0),
            # dict(type='Triplet', margin=0.60, weights=1.0),
        ]
    ),

    # Solver: 学习率调整策略, 从 torch.optim.lr_scheduler 加载
    # lr
    lr_scheduler=dict(type="ExponentialLR", gamma=0.99998),

    # optim
    optimizer=dict(type="Adam", lr=4e-4, weight_decay=1e-5),

    # warm_up
    warm_up=dict(length=2000, min_lr=4e-6, max_lr=4e-4, froze_num_lyers=8),
)


if __name__ == "__main__":
    print(config)
