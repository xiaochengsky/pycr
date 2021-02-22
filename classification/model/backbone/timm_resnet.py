import torch
import torch.nn as nn
from .timm.models import resnet as timm_resnet
from .timm.models import resnest as timm_resnest
from .timm.models import res2net as timm_res2net


# 2048
class Res2Net50(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(Res2Net50, self).__init__()
        self.model = timm_res2net.res2net50_26w_8s(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class Res2Net101(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(Res2Net101, self).__init__()
        self.model = timm_res2net.res2net101_26w_4s(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class ResNeSt101(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(ResNeSt101, self).__init__()
        self.model = timm_resnest.resnest101e(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class ResNeSt200(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(ResNeSt200, self).__init__()
        self.model = timm_resnest.resnest200e(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class ResNeSt269(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(ResNeSt269, self).__init__()
        self.model = timm_resnest.resnest269e(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


if __name__ == '__main__':
    inputs = torch.randn(2, 3, 224, 224)
    model = ResNeSt269(pretrained=False)
    print(model(inputs).shape)

