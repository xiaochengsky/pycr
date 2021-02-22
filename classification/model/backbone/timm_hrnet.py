import torch
import torch.nn as nn
from .timm.models import hrnet as timm


# 2048
class HRNet_w40(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(HRNet_w40, self).__init__()
        self.model = timm.hrnet_w40(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class HRNet_w48(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(HRNet_w48, self).__init__()
        self.model = timm.hrnet_w48(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class HRNet_w64(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(HRNet_w64, self).__init__()
        self.model = timm.hrnet_w64(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = HRNet_w64(pretrained=False)
    print(model(inputs).shape)

