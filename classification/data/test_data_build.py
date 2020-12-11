import sys

# print(sys.path.remove('D:\\PyCharmProjects\\Cassava\\features\\data'))
# print(sys.path)
from PIL import Image
from torchvision import transforms
import cv2
from .dataloader.build import create_dataloader


# if __name__ == '__main__':
train_pipeline = dict(
    dataloader=dict(
        batch_size=1,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        # collate_fn="my_collate_fn",
    ),

    dataset=dict(
        type="train_dataset",
        root_dir=r"D:\PyCharmProjects\datasets\cassava-leaf-disease-classification",
        # images_per_classes=4,
        # classes_per_minibatch=1,
    ),

    transforms=[
        # dict(type="ShiftScaleRotate", p=0.3, shift_limit=0.1, scale_limit=(-0.5, 0.2), rotate_limit=15),
        # dict(type="IAAPerspective", p=0.1, scale=(0.05, 0.15)),
        # dict(type="ChannelShuffle", p=0.1),
        # dict(type="RandomRotate90", p=0.2),
        # dict(type="RandomHorizontalFlip", p=0.5),
        dict(type="ToTensor", ),
        # dict(type="RandomVerticalFlip", p=0.5),
        # dict(type="ColorJitter", brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # dict(type="RandomErasing", p=0.2),
        # dict(type="RandomPatch", p=0.05, pool_capacity=1000, min_sample_size=100, patch_min_area=0.01,
        #      patch_max_area=0.2, patch_min_ratio=0.2, p_rotate=0.5, p_flip_left_right=0.5),

        # dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
    ],
)


train_dataloader = create_dataloader(train_pipeline)
print(len(train_dataloader))
for image, target, image_name in train_dataloader:
    print(image_name)
    print(image)
    print(target)
    # image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    image = transforms.ToPILImage()(image.squeeze(0))
    print(len(image.split()))
    # image.show()
    # cv2.imshow('xx', image)

    im = cv2.imread(r"D:/PyCharmProjects/datasets\cassava-leaf-disease-classification\train_images\2671853400.jpg")
    print(im.shape)
    cv2.imshow('raw', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)

