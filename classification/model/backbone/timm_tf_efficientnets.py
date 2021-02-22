import torch
import torch.nn as nn
from .timm import create_model


# 1536
class timm_efficient_b3(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(timm_efficient_b3, self).__init__()
        self.model = create_model('tf_efficientnet_b3_ns', pretrained=pretrained, num_classes=num_classes)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 1792
class timm_efficient_b4(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(timm_efficient_b4, self).__init__()
        self.model = create_model("tf_efficientnet_b4_ns", pretrained=pretrained, num_classes=num_classes)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2048
class timm_efficient_b5(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(timm_efficient_b5, self).__init__()
        self.model = create_model("tf_efficientnet_b5_ns", pretrained=pretrained, num_classes=num_classes)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2304
class timm_efficient_b6(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(timm_efficient_b6, self).__init__()
        self.model = create_model("tf_efficientnet_b6_ns", pretrained=pretrained, num_classes=num_classes)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


# 2560
class timm_efficient_b7(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(timm_efficient_b7, self).__init__()
        self.model = create_model("tf_efficientnet_b7_ns", pretrained=pretrained, num_classes=num_classes)

    def forward(self, inputs):
        return self.model.forward_features(inputs)


if __name__ == "__main__":
    inputs = torch.randn(1, 3, 224, 224)
    model = timm_efficient_b7(pretrained=False)
    outputs = model(inputs)
    print(outputs.shape)
    length_backbone = len(list(model.named_children()))
    length_model_backbone = len(list(model.model.children()))
    # 1
    print("len_layers : ", length_backbone)
    print("len_model_layers: ", length_model_backbone)
    # state_dict = model.state_dict()
    # for key in state_dict.keys():
    #     print(key)

