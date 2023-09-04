import os

import torch.nn as nn

__all__ = ['repvit_m1']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


from timm.models.layers import SqueezeExcite

import torch


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


from timm.models.vision_transformer import trunc_normal_


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier


class RepViT(nn.Module):
    def __init__(self, cfgs, num_classes=1000, distillation=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                                          Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(output_channel, num_classes, distillation)

    def forward(self, x):
        # x = self.features(x)
        for f in self.features:
            x = f(x)
            print(x.shape)
        print('enter pool')
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return x


from timm.models import register_model


@register_model
def repvit_m1(pretrained=False, num_classes=1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3, 2, 48, 1, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 48, 0, 0, 1],
        [3, 2, 96, 0, 0, 2],
        [3, 2, 96, 1, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 96, 0, 0, 1],
        [3, 2, 192, 0, 1, 2],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 1, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 192, 0, 1, 1],
        [3, 2, 384, 0, 1, 2],
        [3, 2, 384, 1, 1, 1],
        [3, 2, 384, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m2(pretrained=False, num_classes=1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m3(pretrained=False, num_classes=1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


class RepVitNet(nn.Module):
    def __init__(self, num_classes=6, variant='m1', pretrained=True,
                 pretrained_path='/home/ycc/Downloads/repvit_m1_distill_300.pth'):
        super(RepVitNet, self).__init__()
        self.num_classes = num_classes
        self.variant = variant
        # self.model = nn.ModuleList(
        #     *list(eval(f'repvit_{variant}')(pretrained=pretrained, num_classes=num_classes,
        #                                     distillation=True).children())[:-1])
        self.model = eval(f'repvit_{variant}')(pretrained=pretrained, num_classes=num_classes, distillation=True)
        self.model.classifier = nn.Identity()

        if pretrained:
            print('will pretrained RepVitNet')
            if not os.path.exists(pretrained_path):
                pass
            pretrain_model = torch.load(pretrained_path)['model']
            pretrain_model_dict = dict(pretrain_model.items())
            self.model_dict = dict(self.model.state_dict())
            # for k, v in self.model_dict.items():
            #     self.model_dict[k] = pretrain_model_dict[]

    def forward(self, x):
        out = self.model(x)
        return out


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


if __name__ == '__main__':
    # model = repvit_m1()
    #
    # input_shape = (1, 3, 224, 224)
    # dummy_input = torch.randn(input_shape)
    # model.eval()
    # print(model(dummy_input).sum())
    #
    # exit(0)
    torch.random.manual_seed(0)
    model = RepVitNet(num_classes=6, variant='m1', pretrained=False)
    torch.save(model, 'model.pth')
    replace_batchnorm(model)
    torch.save(model, 're_model.pth')

    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape)
    print(model(dummy_input).sum())

    layer = 0
    model_dict = dict(model.state_dict())
    # for k, v in dict(model.state_dict()).items():
    #     layer += 1
    #     print(k, v.shape)
    #
    # print(layer)

    pre_model = torch.load('/home/ycc/Downloads/repvit_m1_distill_300.pth')['model']
    pre_model_dict = dict(pre_model.items())
    for k, v in model_dict.items():
        layer += 1
        print(k, v.shape)
        p_v = model_dict[k].sum()
        p_k = k.replace('model.', '')
        model_dict[k] = pre_model_dict[p_k]
        l_v = model_dict[k].sum()
        print(p_v, l_v)
    print(layer)
