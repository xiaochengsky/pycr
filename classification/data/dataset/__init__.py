
# 构建 dataset 接口
from . import build as datasets


def build_dataset(cfg_dataset, transforms):
    cfg_dataset = cfg_dataset.copy()

    # train_dataset or test_dataset
    dataset_type = cfg_dataset.pop("type")

    # params
    dataset_params = cfg_dataset
    print()
    if hasattr(datasets, dataset_type):
        # 返回对应的类别 train_dataset / test_dataset / val_dataset
        dataset = getattr(datasets, dataset_type)(**dataset_params, transforms=transforms)
    else:
        raise ValueError("\'type\'{} of dataset is not defined!!!".format(dataset_type))
    return dataset

