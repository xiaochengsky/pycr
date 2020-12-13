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
        sample = {'image': image}
        if self.transforms:
            sample = self.transforms(sample)
        image = sample['image']
        return image, image_name


class train_dataset_triplet(Dataset):
    def __init__(self, root_dir, images_per_classes, classes_per_minibatch, transforms):
        self.transforms = transforms
        self.classes_per_minibatch = classes_per_minibatch
        self.images_per_classes = images_per_classes
        self.minibatch = self.classes_per_minibatch * self.images_per_classes

        self.image_path = os.path.join(root_dir, "img_path")
        self.images = os.listdir(self.image_path)
        self.num_all_classes = 5

        label_name = 'label.txt'
        # [] [] [] [] []
        # 每个类的image_name
        self.datas = [[] for i in range(self.num_all_classes)]
        self.labels = []

        file = open(os.path.join(root_dir, label_name))
        while True:
            word = file.readline()
            if not word:
                break
            word = word.rsplit('\n').split(',')
            label = int(word[1])
            image_name = word[0]
            self.datas[label].append(image_name)
            self.labels.append(label)
        file.close()

        self.steps_all = int(len(self.data) / classes_per_minibatch)

        # 类别乱序
        self.read_order = random.sample(random(0, self.num_all_classes), self.num_all_classes)

    def shuffle(self):
        self.read_order = random.sample(random(0, self.num_all_classes), self.num_all_classes)

    def __len__(self):
        return

    def get_item(self, class_id, img_id):
        img = cv2.imread(self.data[class_id][img_id])
        sample = {"image": img}
        if self.transforms:
            sample = self.transforms(sample)
        img = sample["image"]

        img = img.unsqueeze(0)
        return img

    # 获取第 step 个 minibatch
    # def __getitem__(self, step):
    #     if step > self


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
        sample = {"image": image}
        if self.transform:
            sample = self.transform(sample)
        image = sample["image"]
        return image, image_name
