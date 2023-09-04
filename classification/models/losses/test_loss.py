# -*- coding: utf-8 -*-
# @Time : 2022/12/8 上午11:50
# @Author : XXX
# @File : test_loss.py

import torch
import torch.nn as nn

if __name__ == '__main__':
    labels = torch.FloatTensor([[1, 1, 45], [1, 0, 45]])
    preds = torch.FloatTensor([[1, 1, 1], [1, 1, 1]])
    mask = labels != 45
    print(mask)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(preds, labels)
    print(loss)
    loss = loss * mask
    print(loss)
    print(loss.sum())
    print(loss.mean())
    