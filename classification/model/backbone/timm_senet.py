import torch
import torch.nn as nn
from .timm.models import senet as timm


# 2048
class SEResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(SEResNet50, self).__init__()
        self.model = timm.legacy_seresnet50(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class SEResNet101(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(SEResNet101, self).__init__()
        self.model = timm.legacy_seresnet101(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class SEResNeXt50(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(SEResNeXt50, self).__init__()
        self.model = timm.legacy_seresnext50_32x4d(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class SEResNeXt101(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(SEResNeXt101, self).__init__()
        self.model = timm.legacy_seresnext101_32x4d(pretrained=pretrained)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


if __name__ == '__main__':
    inputs = torch.randn(1, 3, 224, 224)
    model = SEResNeXt101(pretrained=False)
    print(model(inputs).shape)
