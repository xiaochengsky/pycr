# 定义抽取特征的方式，包括：
## 1. 读取数据的增强
## 2. 抽取的特征层
## 3. 保存的特征名

config = dict(
    # Basic cofnfig
    enable_backends_cudnn_benchmark=True,
    max_epochs=100 + 1,

    # 间隔多少 iter 打印一次 loss, acc 等数据
    log_periods=5,

    # features存储: features/tag/'
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
    max_num_devices=1,

    # DataLoader Config
        ## dataloader: image2batch, 继承自 torch.utils.data.DataLoader
        ## dataset: 加载 image 和 label: data/dataset/bulid.py
        ## transforms: 在线数据增强: data/transforms/opencv_transforms.py
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
            dict(type="RescalePad", output_size=640),
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
        net=dict(type='ResNet50_GeM_Identity_CE', ),
        backbone=dict(type='resnet50', pretrained=True),
        aggregation=dict(type="GeneralizedMeanPooling", ),
        heads=dict(type='IdentityHead', ),
        losses=[
            dict(type='CrossEntroy', in_feat=2048, num_classes=5, weight=1.0)
            # dict(type='Arcface', in_feat=2048, num_classes=5, scale=35, margin=0.30, drop_out=0.2, weights=1.0),
            # dict(type='Triplet', margin=0.60, weights=1.0),
        ]
    ),

    extract_pipeline=dict(
        #
        assemble=0,

        extractor=dict(
            # 从 backbone 的输出抽取特征
            extractor_type="before"
        ),

        aggregator=dict(
            # 聚合特征的方式
            aggregator_type="GeM"
        ),

        # model + extractor_type + dims
        save_dirs=dict(
            dir="features",
        ),
    ),

)


if __name__ == "__main__":
    print(config)
