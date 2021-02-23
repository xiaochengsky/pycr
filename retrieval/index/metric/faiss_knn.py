import faiss
import numpy as np


class faissKnn:
    def __init__(self, dim='512'):
        super(faissKnn, self).__init__()
        self.dim = dim

    def calc_knn(self, type='cosine', gallery_vector='', query_vector='', k=10):
        if type == 'cosine':
            index = faiss.IndexFlatIP(self.dim)
            faiss.normalize_L2(gallery_vector)
            faiss.normalize_L2(query_vector)
        else:
            index = faiss.IndexFlatL2(self.dim)

        index.train(gallery_vector)
        index.add(gallery_vector)
        D, I = index.search(query_vector, k)
        return D, I

    def __call__(self, type='cosine', gallery_vector=[], query_vector=[], k=10):
        return self.calc_knn(type, gallery_vector, query_vector, k)


if __name__ == '__main__':
    fk = faissKnn(3)
    q = np.array([[1, 1, 1], [1, 1, 3]], dtype=np.float32)
    g = np.array([[1, 1, 2], [1, 1, 3]], dtype=np.float32)

    dis, idx = fk('cosine', g, q, 2)
    print(dis)
    print(idx)
