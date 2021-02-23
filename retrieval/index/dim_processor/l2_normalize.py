import numpy as np
from sklearn.preprocessing import normalize


class L2Normalize:
    """
    L2 normalize the features.
    """
    default_hyper_params = dict()

    def __init__(self):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(L2Normalize, self).__init__()

    def __call__(self, fea: np.ndarray) -> np.ndarray:
        return normalize(fea, norm="l2")


if __name__ == '__main__':
    l2 = L2Normalize()

    import torch
    v = []
    for i in range(3):
        a = torch.Tensor([[1, 2, 3], [4, 5, 6]]).view(-1, 3).numpy()
        # print(a.shape)
        v.append(a)

    v = np.concatenate(v[:], axis=1)
    print(v)
    v = l2(v)
    print(v)
