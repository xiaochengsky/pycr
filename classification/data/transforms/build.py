# from . import cv2_transforms as transforms
# import torchvision.transforms as transforms


def build_transforms(cfg_transforms):
    cfg_transforms = cfg_transforms.copy()
    transforms_list = list()
    for item in cfg_transforms:
        transforms_type = item.pop("type")

        if transforms_type == 'transforms_type':
            if item['backend'] == 'cv2':
                from . import cv2_transforms as transforms
            else:
                # import torchvision.transforms as transforms
                from . import transforms as transforms
        else:
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

    # print(transforms_list)
    return transforms.Compose(transforms_list)
