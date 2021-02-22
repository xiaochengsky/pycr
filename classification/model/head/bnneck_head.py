import torch
import torch.nn as nn
from ...utils import weights_init_kaiming
import torch.autograd as autograd


class BNneckHead(nn.Module):
    def __init__(self, in_feat, num_classes):
        super(BNneckHead, self).__init__()
        self.bnneck = nn.BatchNorm2d(in_feat)
        self.bnneck.apply(weights_init_kaiming)
        self.bnneck.bias.requires_grad_(False)

    def forward(self, features):
        # [N, C, 1, 1](来自于 aggregation 模块) -> [N, C]
        return self.bnneck(features)[..., 0, 0]


class BNneckHead_Dropout(nn.Module):
    def __init__(self, in_feat, num_classes, dropout_rate=0.15):
        super(BNneckHead_Dropout, self).__init__()
        self.bnneck = nn.BatchNorm2d(in_feat)
        # self.bnneck.apply(weights_init_kaiming)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features):
        return self.dropout(self.bnneck(features)[..., 0, 0])


if __name__ == '__main__':
    input = autograd.Variable(torch.randn(3, 3, 1, 1))
    m = BNneckHead(3, 3)
    res = m(input)
    print(res)

    n = BNneckHead_Dropout(3, 3, dropout_rate=1)
    rel = n(input)
    print(rel)
