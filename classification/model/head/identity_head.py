import torch
import torch.nn as nn


class IdentityHead(nn.Module):
    def __init__(self):
        super(IdentityHead, self).__init__()

    def forward(self, features):
        return features[..., 0, 0]


if __name__ == '__main__':
    n = torch.randn(3, 3, 1, 1)
    m = IdentityHead()
    print(m(n).shape)
