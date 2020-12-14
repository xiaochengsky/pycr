from glob import glob
import os
import torch
from PIL import Image, ImageFile
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = 1
ROOT_DIR = '/opt/kaggle/Cassava-leaf-dataset'
TRAIN_LABEL = 'train_labels.txt'
VAL_LABEL = 'val_labels.txt'

transform = transforms.Compose([transforms.Resize(512),
                                # transforms.CenterCrop(512),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
                                ])


class train_dataset(Dataset):
    def __init__(self, root_dir, label_path, transforms):
        self.root_dir = root_dir
        self.transform = transforms
        self.labels = []
        self.datas = []
        # label_name = 'train_labels.txt'
        label_name = label_path

        file = open(os.path.join(root_dir, label_name))
        while True:
            word = file.readline()
            if not word:
                break
            word = word.split(',')
            label = int(word[1])
            image_name = word[0]
            self.datas.append(image_name)
            self.labels.append(label)
        file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name, target = self.datas[idx], self.labels[idx]
        # image = cv2.imread(os.path.join(self.root_dir, image_name))
        # image = image[:, :, ::-1].copy()
        # sample = {"image": image, "target": target}
        # if self.transform:
        #     sample = self.transform(sample)
        # image, target = sample["image"], sample["target"]

        image = Image.open(os.path.join(self.root_dir, image_name))
        if self.transform:
            image = self.transform(image)
        return image, target, image_name


class val_dataset(Dataset):
    def __init__(self, root_dir, label_path, transforms):
        self.root_dir = root_dir
        self.transform = transforms
        self.labels = []
        self.datas = []
        # label_name = 'val_labels.txt'
        label_name = label_path

        file = open(os.path.join(root_dir, label_name))
        while True:
            word = file.readline()
            if not word:
                break
            word = word.split(',')
            label = int(word[1])
            image_name = word[0]
            self.datas.append(image_name)
            self.labels.append(label)
        file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name, target = self.datas[idx], self.labels[idx]
        # image = cv2.imread(os.path.join(self.root_dir, image_name))
        # sample = {"image": image, "target": target}
        # if self.transform:
        #     sample = self.transform(sample)
        # image, target = sample["image"], sample["target"]
        # return image, target, image_name
        image = Image.open(os.path.join(self.root_dir, image_name))
        if self.transform:
            image = self.transform(image)
        return image, target, image_name



train_dataset = train_dataset(ROOT_DIR, TRAIN_LABEL, transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

test_dataset = val_dataset(ROOT_DIR, VAL_LABEL, transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

mean = torch.zeros(3)
std = torch.zeros(3)

i = 0
print('train total: ', len(train_loader))
for X, _ in train_loader:
    i += 1
    for d in range(3):
        mean[d] += X[:, d, :, :].mean()
        std[d] += X[:, d, :, :].std()
    if i % 10000 == 0:
        print('loaded 1w+...')

i = 0
print('test total: ', len(test_loader))
for X, _ in test_loader:
    i += 1
    for d in range(3):
        mean[d] += X[:, d, :, :].mean()
        std[d] += X[:, d, :, :].std()
    if i % 10000 == 0:
        print('loaded 1w+...')

mean.div_(len(train_loader) + len(test_loader))
std.div_(len(train_loader) + len(test_loader))

print(list(mean.numpy()), list(std.numpy()))
