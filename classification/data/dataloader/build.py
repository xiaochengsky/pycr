from ..dataset import build_dataset
from ..transforms.build import build_transforms
# import .build_dataset()

from torch.utils.data import DataLoader


def create_dataloader(cfg_data_pipeline):
    cfg_data_pipeline_c = cfg_data_pipeline.copy()
    cfg_dataset = cfg_data_pipeline_c.pop("dataset")
    cfg_transforms = cfg_data_pipeline_c.pop("transforms")
    cfg_dataloader = cfg_data_pipeline_c.pop("dataloader")

    transforms = build_transforms(cfg_transforms)
    dataset = build_dataset(cfg_dataset, transforms)

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
            dataloader = DataLoader(dataset, **cfg_dataloader)
            return dataloader
