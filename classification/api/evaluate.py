# -*- coding: utf-8 -*-
# @Time : 2022/12/6 上午9:44
# @Author : YANG.C
# @File : evaluate.py
import torch
from tqdm import tqdm
import numpy as np
from easydict import EasyDict


def evaluate_fairness(val_loader, model, cfg):
    model.eval()
    nbv = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=nbv)
    device = cfg.device

    # fairness_score, acc_score,
    correct_num, total_num = 0, 0
    group_ids = [6, 7, 8, 10]
    group_correct_num = {6: 0, 7: 0, 8: 0, 10: 0}
    group_total_num = {6: 0, 7: 0, 8: 0, 10: 0}

    for i, batch in pbar:
        imgs = batch['image']
        imgs = imgs.to(device, non_blocking=True).float()
        targets = batch['gt_label']
        targets = targets.to(device)
        groups = batch['gt_group']
        preds = model(imgs, return_loss=False)
        logits = preds['classes']
        preds = logits.argmax(dim=1)
        result = torch.eq(preds, targets)
        correct_num += result.sum().float().item()
        total_num += len(targets)

        for gi in group_ids:
            group_mask = (groups == gi)
            group_correct_num[gi] += (group_mask * result.cpu().numpy()).sum().float().item()
            group_total_num[gi] += group_mask.sum()

    minority_acc = group_correct_num[10] / group_total_num[10]
    abs_difference = 0
    for k in group_correct_num.keys():
        abs_difference += abs((group_correct_num[k] / group_total_num[k]) - minority_acc)

    spd = abs_difference / len(group_correct_num)
    fairness_score = ((0.2 - spd) / 0.2) / 3
    overall_acc = correct_num / total_num
    acc_score = overall_acc / 3
    performance_score = acc_score + fairness_score
    return {
        'acc_score': acc_score,
        'overall_score': overall_acc,
        'fairness_score': fairness_score,
        'performance_score': performance_score,
    }


def evaluate(val_loader, model, cfg):
    model.eval()
    nbv = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=nbv)
    device = cfg.device

    correct, total = 0, 0
    for i, batch in pbar:
        imgs = batch['image']
        imgs = imgs.to(device, non_blocking=True).float()
        targets = batch['gt_label']
        targets = targets.to(device)
        preds = model(imgs, return_loss=False)
        logits = preds['classes']

        # top1
        pred = logits.argmax(dim=1)
        correct += torch.eq(pred, targets).sum().float().item()
        total += len(targets)

    acc = correct / total

    return {
        'acc': acc
    }


def evaluate_splitfc(val_loader, model, cfg):
    model.eval()
    nbv = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=nbv)
    device = cfg.device
    correct1, correct2, total = 0, 0, 0

    for i, batch in pbar:
        imgs = batch['image']
        imgs = imgs.to(device, non_blocking=True).float()
        targets = batch['gt_label']
        targets = targets.to(device)
        preds = model(imgs, return_loss=False)
        logits = preds['classes']
        logit1, logit2 = logits

        pred1 = logit1.argmax(dim=1)  # [N, 7]
        pred2 = logit2.argmax(dim=1)  # [N, 15]

        mask = targets < 7
        correct1 += torch.eq(pred1[mask], targets[mask]).sum().float().item()
        correct2 += torch.eq(pred2[(~mask)], targets[~mask] - 7).sum().float().item()
        total += len(targets)

    acc = (correct1 + correct2) / total
    return {
        'correct1': correct1,
        'correct2': correct2,
        'acc': acc,
    }


def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.5, index=None, cfg=None):
    """
    index: evaluated label index
    """
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    if index is not None:
        pred_label = pred_label[:, index]
        gt_label = gt_label[:, index]

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    # instance_f1 = np.mean(instance_f1)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result


def evaluate_mulitlabel(val_loader, model, cfg):
    model.eval()
    nbv = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=nbv)
    device = cfg.device

    correct, total = 0, 0
    ma_preds, ma_labels = [], []
    for i, batch in pbar:
        imgs = batch['image']
        imgs = imgs.to(device, non_blocking=True).float()
        targets = batch['gt_label']
        targets = targets.to(device)
        logits = model(imgs)
        logits = logits['classes']

        # evaluate per image
        for i in range(len(logits)):
            preds = torch.sigmoid(logits[i]) >= 0.5
            mask = targets[i] != 45
            if (preds * mask == targets[i] * mask).sum() == len(targets[i]):
                correct += 1

            # for mA
            ma_preds.append((preds * mask).tolist())
            ma_labels.append((targets[i] * mask).tolist())

        # top1
        total += len(targets)

    ma_preds = torch.tensor(ma_preds)
    ma_labels = torch.tensor(ma_labels)
    classes = ma_preds.shape[2]
    ma_preds = ma_preds.view(-1, classes).cpu().numpy()
    ma_labels = ma_labels.view(-1, classes).cpu().numpy()
    ma_info = get_pedestrian_metrics(ma_labels, ma_preds)

    top1 = correct / total

    return {
        'acc': ma_info.instance_acc,
        'top1': top1,
        'ma': ma_info,
    }

# class Evaluate():
#     def __init__(self):
