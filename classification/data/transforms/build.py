from . import cv_transforms as transforms


def build_transforms(cfg_transforms):
    cfg_transforms = cfg_transforms.copy()
    transforms_list = list()
    for item in cfg_transforms:
        transforms_type = item.pop("type")
        transforms_params = item
        # for debug
        # print(transforms_params)
        if hasattr(transforms, transforms_type):
            # for debug
            # print('transforms: ', transforms)
            # print(getattr(transforms, transforms_type))
            transforms_list.append(getattr(transforms, transforms_type)(**transforms_params))
        else:
            raise ValueError("\'type\'{} of transforms is not defined!!!".format(transforms_type))

    # for debug
    # print(transforms_list)
    return transforms.Compose(transforms_list)
