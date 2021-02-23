# -*- coding: utf-8 -*-

import torch
import numpy as np
import faiss


class re_faissKnn_:
    def __init__(self, dim='512'):
        super(re_faissKnn_, self).__init__()
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


class QE:

    def __init__(self, hps):
        super(QE, self).__init__()
        self._hyper_params = hps
        self.knn = re_faissKnn_(dim=self._hyper_params["dim"])

    def __call__(self, query_fea=[], gallery_fea=[]):
        type = self._hyper_params['type']
        weight = self._hyper_params['weight']
        dis, sorted_index = self.knn(type, gallery_fea, query_fea, k=self._hyper_params['k'])
        # print(dis)
        # sorted_index = np.argsort(dis, axis=1)
        # print(sorted_index)
        for i in range(self._hyper_params['qe_times']):
            sorted_index = sorted_index[:, :self._hyper_params['qe_k']]
            # print(sorted_index)
            sorted_index = sorted_index.reshape(-1)
            # print(sorted_index, sorted_index.shape)
            # print(gallery_fea[sorted_index])
            if not weight:
                # print(gallery_fea[sorted_index].reshape(query_fea.shape[0], -1, query_fea.shape[1]))
                requery_fea = gallery_fea[sorted_index].reshape(query_fea.shape[0], -1, query_fea.shape[1]).sum(axis=1)
            else:
                tmp = gallery_fea[sorted_index].reshape(query_fea.shape[0], -1, query_fea.shape[1])
                weights = np.arange(1, tmp.shape[1] + 1)
                weights = np.expand_dims(weights, axis=0)
                weights = np.expand_dims(weights, axis=2)
                weights = 1 / weights
                requery_fea = (tmp * weights).sum(axis=1)
                # requery_fea = gallery_fea[sorted_index].reshape(query_fea.shape[0], -1, query_fea.shape[1]).sum(axis=1)
            # print(requery_fea)
            requery_fea = requery_fea + query_fea
            # print(requery_fea)
            query_fea = requery_fea
            # print(query_fea)

        return query_fea / (self._hyper_params['qe_k'] + 1)


if __name__ == '__main__':
    # qe: 执行次数, 这个就设置为1次
    # qe_k: 扩展查询的个数(一般为3)
    # dim: 检索距离时的维度
    # weight: 是否用加权平均(False, 全部求平均　　True: 加权平均)
    # k: 检索 topk 个,
    # type: 检索的距离度量('cosine', 'L2')

    default_hyper_params = {
        "qe_times": 1,
        "qe_k": 3,
        "dim": 3,
        "weight": False,
        "k": 10,
        "type": 'cosine',
    }
    q = np.array([[1, 1, 1], [1, 1, 9]], dtype=np.float32)
    g = np.array([[1, 1, 2], [1, 1, 3], [1, 1, 9], [2, 2, 6], [3, 3, 3]], dtype=np.float32)
    qe = QE(default_hyper_params)

    # 传入 query, gallery 特征, 得到新的 query 特征
    query = qe(q, g)
    print(query)




