# -*- coding: utf-8 -*-

import torch
import numpy as np
import faiss


class _faissKnn:
    def __init__(self, dim='512'):
        super(_faissKnn, self).__init__()
        self.dim = dim

    def calc_knn(self, type='cosine', gallery_vector='', query_vector='', k=10):
        # copy()
        gallery_enhance = gallery_vector.copy()
        query_enhance = query_vector.copy()
        if type == 'cosine':
            # L2 norm
            index = faiss.IndexFlatIP(self.dim)
            faiss.normalize_L2(gallery_enhance)
            faiss.normalize_L2(query_enhance)
        else:
            index = faiss.IndexFlatL2(self.dim)

        # faiss index
        index.train(gallery_enhance)
        index.add(gallery_enhance)
        D, I = index.search(query_enhance, k)
        return D, I

    def __call__(self, type='cosine', gallery_vector=[], query_vector=[], k=10):
        return self.calc_knn(type, gallery_vector, query_vector, k)


class DBA:
    """
    Every feature in the database is replaced with a weighted sum of the point ’s own value and those of its top k nearest neighbors (k-NN).
    c.f. https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

    Hyper-Params:
        enhance_k (int): number of the nearest points to be calculated.
    """
    default_hyper_params = {
        "enhance_k": 10,
        "dim": 512,
    }

    def __init__(self, hps):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(DBA, self).__init__()
        self._hyper_params = hps
        # self.knn = KNN(knn_hps)
        self.knn = _faissKnn(dim=self._hyper_params["dim"])

    def __call__(self, feature, type):
        # 在 gallery 中, 寻找每个图片中距离最近的图片 top_k(3) 张图片
        # sorted_idx: [5, 5], sorted_idx: [5, 3]

        _, sorted_idx = self.knn(type=type, gallery_vector=feature, query_vector=feature,
                                 k=self._hyper_params["enhance_k"] + 1)
        # 除去图片自身, sorted_idx: [5, 2]
        sorted_idx = sorted_idx[:, 1:].reshape(-1)

        # 对每个图片的特征和与其相近的 topk-1 个图片的特征做 sum
        # arg_fea = feature[sorted_idx].view(feature.shape[0], -1, feature.shape[1]).sum(dim=1)
        arg_fea = feature[sorted_idx].reshape(feature.shape[0], -1, feature.shape[1]).sum(axis=1)
        feature = feature + arg_fea

        # l2 正则归一化
        # feature = feature / torch.norm(feature, dim=1, keepdim=True)
        norm = np.linalg.norm(feature, axis=1, keepdims=True)
        feature = feature / norm
        return feature


if __name__ == '__main__':
    default_hyper_params = {
        "enhance_k": 2,
        'dim': 512,
    }
    # knn = KNN(default_hyper_params)
    # q = torch.Tensor([[1, 1, 1], [1, 1, 9]])
    # g = torch.Tensor([[1, 1, 2], [1, 1, 3], [1, 1, 9], [2, 2, 6], [3, 3, 3]])
    q = np.array([[1, 1, 1], [1, 1, 9]])
    g = np.array([[1, 1, 2], [1, 1, 3], [1, 1, 9], [2, 2, 6], [3, 3, 3]])
    # dis = knn._cal_dis(q, g)
    #
    # dis, sorted_idx = knn(g, g)
    # print('dis:')
    # print(dis)
    # print('argsort:')
    # sorted_idx = sorted_idx[:, 1:].reshape(-1)
    # print(sorted_idx, sorted_idx.shape)
    #
    # # arg_fea = feature[sorted_idx].view(feature.shape[0], -1, feature.shape[1]).sum(dim=1)
    # arg_fea = g[sorted_idx]
    # print(arg_fea, arg_fea.shape)
    #
    # arg_fea = g[sorted_idx].view(g.shape[0], -1, g.shape[1])
    # print(arg_fea, arg_fea.shape)
    #
    # arg_fea = g[sorted_idx].view(g.shape[0], -1, g.shape[1]).sum(dim=1)
    # print(arg_fea, arg_fea.shape)
    #
    # g = g + arg_fea
    # print(g)
    #
    # feature = g / torch.norm(g, dim=1, keepdim=True)
    # print(feature)
    #     tensor([[0.4683, 0.4683, 0.7493],
    #         [0.4683, 0.4683, 0.7493],
    #         [0.2120, 0.2120, 0.9540],
    #         [0.2120, 0.2120, 0.9540],
    #         [0.4683, 0.4683, 0.7493]])
    dba = DBA(default_hyper_params)
    print(dba(g))
    #     tensor([[0.4683, 0.4683, 0.7493],
    #         [0.4683, 0.4683, 0.7493],
    #         [0.2120, 0.2120, 0.9540],
    #         [0.2120, 0.2120, 0.9540],
    #         [0.4683, 0.4683, 0.7493]])
