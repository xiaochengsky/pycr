import torch
import torch.nn as nn
from .timm.models import regnet as timm


# 1512
class RegNetY_032(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(RegNetY_032, self).__init__()
        self.model = timm.regnety_032(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2240
class RegNetY_120(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(RegNetY_120, self).__init__()
        self.model = timm.regnety_120(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 3712
class RegNetY_320(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(RegNetY_320, self).__init__()
        self.model = timm.regnety_320(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = RegNetY_120(pretrained=False)
    print(model.eval())
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model(inputs).shape)
    print('Trainable: ', trainable_num)
    print('Total: ', total_num)
