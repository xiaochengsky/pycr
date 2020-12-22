from ..dataset import build_dataset
from ..transforms.build import build_transforms
# import .build_dataset
from . import sampler as sampler
from torch.utils.data import DataLoader


def build_sampler(cfg_sampler, dataset):
    cfg_sampler_c = cfg_sampler.copy()
    sampler_type = cfg_sampler_c.pop("type")
    sampler_params = cfg_sampler_c
    if hasattr(sampler, sampler_type):
        dataset_sampler = getattr(sampler, sampler_type)(dataset)
    else:
        raise ValueError("\'type\'{} of dataset is not defined!!!".format(sampler_type))
    return dataset_sampler


def create_dataloader(cfg_data_pipeline):
    cfg_data_pipeline_c = cfg_data_pipeline.copy()
    cfg_dataset = cfg_data_pipeline_c.pop("dataset")
    cfg_transforms = cfg_data_pipeline_c.pop("transforms")
    cfg_dataloader = cfg_data_pipeline_c.pop("dataloader")

    transforms = build_transforms(cfg_transforms)
    dataset = build_dataset(cfg_dataset, transforms)

    if "sampler" in cfg_data_pipeline_c:
        cfg_sampler = cfg_data_pipeline_c.pop("sampler")
        dataset_sampler = build_sampler(cfg_sampler, dataset)
        print("sampler")

    if "sampler" in cfg_dataloader:
        cfg_sampler = cfg_dataloader.pop("sampler")
        pass
    else:
        if "collate_fn" in cfg_dataloader:
            cfg_collate_fn = cfg_dataloader.pop("collate_fn")
            if hasattr(my_collate_fn, cfg_collate_fn):
                collate_fn = getattr(my_collate_fn, cfg_collate_fn)
                dataloader = DataLoader(dataset, collate_fn=collate_fn, **cfg_dataloader)
                return dataloader
        else:
            print("use pytorch collate_fn")
            if "sampler" in cfg_data_pipeline:
                print('use owner sampler')
                dataloader = DataLoader(dataset, sampler=dataset_sampler, **cfg_dataloader)
            else:
                dataloader = DataLoader(dataset, **cfg_dataloader)
            return dataloader

