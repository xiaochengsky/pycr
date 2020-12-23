import sys
import os
import datetime
import copy
import torch
from tqdm import tqdm
import faiss
import numpy as np

from ...classification.data.dataloader.build import create_dataloader
from ...classification.utils.utils import *
from ..configs import load_args, merge_from_arg


gallery_vectors_path = "/home/yc/PycharmProjects/features/extract_r50_gq/gallery_features.npy"
gallery_fns_path = "/home/yc/PycharmProjects/features/extract_r50_gq/gallery_names.npy"
gallery_targets_path = "/home/yc/PycharmProjects/features/extract_r50_gq/gallery_targets.npy"

query_vectors_path = "/home/yc/PycharmProjects/features/extract_r50_gq/query_features.npy"
query_fns_path = "/home/yc/PycharmProjects/features/extract_r50_gq/query_names.npy"
query_targets_path = "/home/yc/PycharmProjects/features/extract_r50_gq/query_targets.npy"

# gallery_vectors = np.load(gallery_vectors_path, allow_pickle=True).astype('float32')
gallery_vectors = np.load(gallery_vectors_path, allow_pickle=True)
gallery_fns = np.load(gallery_fns_path, allow_pickle=True)
gallery_targets = np.load(gallery_targets_path, allow_pickle=True)


query_vectors = np.load(query_vectors_path, allow_pickle=True)
query_fns = np.load(query_fns_path, allow_pickle=True)
query_targets = np.load(query_targets_path, allow_pickle=True)

print('===== index =====')

faiss.normalize_L2(gallery_vectors)
faiss.normalize_L2(query_vectors)
d = 2048
k = 1
index = faiss.IndexFlatL2(d)
D, I = index.search(query_vectors, k)
num_examples, num_acc = 0, 0
for indices, distances, q_fn, q_ts in zip(I, D, query_fns, query_targets):
    if q_ts == gallery_targets[indices].item():
        num_acc += 1
    num_examples += 1

print(num_acc / num_examples)
