from torch.utils.data import Dataset
import cv2
import pandas as pd
import os
import os.path as osp
import numpy as np
import torch
from os.path import join
import random
from collections import defaultdict
from copy import deepcopy
from PIL import Image


class load_images(Dataset):
    def __init__(self, image_dir, transforms):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_list = os.listdir(self.image_dir)

        self.image_list.sort()

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = cv2.imread(os.path.join(self.image_dir, image_name))
        if self.transforms:
            image = self.transforms(image)
        return image, image_name


class train_dataset_triplet(Dataset):
    def __init__(self, root_dir, label_path, images_per_classes, classes_per_minibatch, transforms, backend='cv2'):
        self.root_dir = root_dir
        self.transform = transforms
        self.classes_per_minibatch = classes_per_minibatch
        self.images_per_classes = images_per_classes
        self.minibatch = self.classes_per_minibatch * self.images_per_classes

        self.num_all_classes = 5
        self.backend = backend

        label_name = label_path

        # [] [] [] [] []
        # 每个类的image_name path
        self.datas = [[] for i in range(self.num_all_classes)]
        self.labels = []

        all_nums = 0
        file = open(os.path.join(root_dir, label_name))
        while True:
            word = file.readline()
            if not word:
                break
            word = word.split(',')
            label = int(word[1])
            image_name = word[0]
            self.datas[label].append(image_name)
            self.labels.append(label)
            all_nums += 1
        file.close()

        # 一个 epoch 要取 steps_all 次
        # self.steps_all = int(len(self.data) / classes_per_minibatch)
        self.steps_all = int(all_nums / self.minibatch)

        # 类别保持乱序
        self.read_order = random.sample(range(0, self.num_all_classes), self.num_all_classes)

    def shuffle(self):
        # 手动打乱
        self.read_order = random.sample(range(0, self.num_all_classes), self.num_all_classes)

    def __len__(self):
        return self.steps_all

    def get_item(self, class_id, img_id):
        image_name, target = self.datas[class_id][img_id], class_id

        if self.backend == 'cv2':
            image = cv2.imread(os.path.join(self.root_dir, image_name))
            image = image[:, :, ::-1].copy()
        else:
            image = Image.open(os.path.join(self.root_dir, image_name))

        if self.transform:
            image = self.transform(image)

        image = image.unsqueeze(0)
        return image, image_name

    # 获取第 step 个 minibatch
    def __getitem__(self, step):
        if step > self.steps_all - 1:
            print('step_train out of size')
            return

        # 跑完一个 iter 打乱一次顺序
        self.shuffle()
        class_ids = self.read_order

        start = True
        labels = []
        img_names = []

        for class_id in class_ids:
            # 查看每个类别图片的数目
            num = min(self.images_per_classes, len(self.datas[class_id]))
            while num < self.images_per_classes:
                class_id = np.random.choice(5, 1)[0]
                num = len(self.data[class_id])

            # 从该类别中随机选择 images_per_classes 张
            img_ids = np.random.choice(len(self.datas[class_id]), self.images_per_classes)
            for img_id in img_ids:
                img_tmp, image_name = self.get_item(class_id, img_id)
                labels.append(class_id)
                if start:
                    imgs = img_tmp.detach().clone()
                    start = False
                else:
                    imgs = torch.cat((imgs, img_tmp), dim=0)
                img_names.append(image_name)

        labels = torch.tensor(labels)
        labels = labels.int()

        return imgs, labels, img_names


class train_dataset(Dataset):
    def __init__(self, root_dir, label_path, transforms, backend='cv2'):
        self.root_dir = root_dir
        self.transform = transforms
        self.labels = []
        self.datas = []
        self.backend = backend
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
        if self.backend == 'cv2':
            image = cv2.imread(os.path.join(self.root_dir, image_name))
            image = image[:, :, ::-1].copy()
        else:
            image = Image.open(os.path.join(self.root_dir, image_name))
        if self.transform:
            image = self.transform(image)
        return image, target, image_name


class val_dataset(Dataset):
    def __init__(self, root_dir, label_path, transforms, backend='cv2'):
        self.root_dir = root_dir
        self.transform = transforms
        self.labels = []
        self.datas = []
        self.backend = backend
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
        if self.backend == 'cv2':
            image = cv2.imread(os.path.join(self.root_dir, image_name))
            image = image[:, :, ::-1].copy()
        else:
            image = Image.open(os.path.join(self.root_dir, image_name))
        if self.transform:
            image = self.transform(image)
        return image, target, image_name


class test_dataset(Dataset):
    def __init__(self, root_dir, image_per_classes, classes_per_minibatch, transforms):
        self.root_dir = root_dir
        self.transform = transforms
        # self.labels = []
        self.datas = []
        label_name = 'test_label.txt'

        file = open(os.path.join(root_dir, label_name))
        while True:
            word = file.readline()
            if not word:
                break
            word = word.rsplit('\n')
            # label = int(word[1])
            image_name = word
            self.datas.append(image_name)
            # self.labels.append(label)
        file.close()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        image_name = self.datas[idx]
        image = cv2.imread(os.path.join(self.root_dir, image_name))
        if self.transform:
            image = self.transform(image)
        return image, image_name


class gallery_dataset(Dataset):
    def __init__(self, root_dir, label_path, transforms):
        self.root_dir = root_dir
        self.transform = transforms
        self.labels = []
        self.datas = []
        # label_name = 'gallery_labels.txt'
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
        image = Image.open(os.path.join(self.root_dir, image_name))
        if self.transform:
            image = self.transform(image)
        return image, target, image_name


class query_dataset(Dataset):
    def __init__(self, root_dir, label_path, transforms):
        self.root_dir = root_dir
        self.transform = transforms
        self.labels = []
        self.datas = []
        # label_name = 'query_labels.txt'
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
        image = Image.open(os.path.join(self.root_dir, image_name))
        if self.transform:
            image = self.transform(image)
        return image, target, image_name
