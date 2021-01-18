# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from torch.nn import functional as F

import pdb


# ===================
#     RGA Module
# ===================

class RGA_Module(nn.Module):
    # [1, 2048, 32, 32]
    # 2048, 1024,
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA_Module, self).__init__()

        # 2048
        self.in_channel = in_channel
        # 1024
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        print('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

        # 2048 // 8 == 256
        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        # Embedding functions for relation features
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
        if self.use_channel:
            num_channel_c = 1 + self.inter_channel
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_c // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
        # [1, 2048, 32, 32]
        b, c, h, w = x.size()
        # pdb.set_trace()
        if self.use_spatial:
            # spatial attention
            # [1, 256, 32, 32]
            theta_xs = self.theta_spatial(x)
            phi_xs = self.phi_spatial(x)

            # [1, 256, 32 * 32]
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            # [1, 32*32=1024, 256]
            theta_xs = theta_xs.permute(0, 2, 1)

            # [1, 256, 32 * 32]
            phi_xs = phi_xs.view(b, self.inter_channel, -1)

            # attentaion 融合每个 HW 上 channel 的信息
            # [1, 1024, 1024]
            Gs = torch.matmul(theta_xs, phi_xs)

            # [1, 1024, 32, 32]
            # H
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)

            # [1, 1024, 32, 32]
            # W
            Gs_out = Gs.view(b, h * w, h, w)

            # [1, 2048, 32, 32]
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)

            # [1, 128, 32, 32]
            #
            Gs_joint = self.gg_spatial(Gs_joint)

            # [1, 256 ,32, 32]
            g_xs = self.gx_spatial(x)
            # [1, 1, 32, 32]
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            # [1, 129, 32, 32]
            ys = torch.cat((g_xs, Gs_joint), 1)

            # [1, 1, 32, 32]
            # 单个 feature map 上, 每个 HW 点对应一个激活值
            W_ys = self.W_spatial(ys)
            if not self.use_channel:
                out = F.sigmoid(W_ys.expand_as(x)) * x
                return out
            else:
                # W_ys.expand_as(x).shape = [1, 2048, 32, 32]
                # F.sigmoid(W_ys.expand_as(x)).shape = [1, 2048, 32, 32]
                x = F.sigmoid(W_ys.expand_as(x)) * x

        if self.use_channel:
            # channel attention
            # x: [1, 2048, 32, 32]
            # x.view(b, c, -1).permute(0, 2, 1).shape = [1, 1024, 2048],
            # xc.shape: [1, 1024, 2048, 1]
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)

            # [1, 128, 2048, 1] -> [1, 128, 2048] -> [1, 2048, 128]
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
            # [1, 128, 2048]
            phi_xc = self.phi_channel(xc).squeeze(-1)
            # [1, 2048, 2048]
            Gc = torch.matmul(theta_xc, phi_xc)
            # [1, 2048, 2048, 1]
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            # [1, 2048, 2048, 1]
            Gc_out = Gc.unsqueeze(-1)

            # [1, 4096, 2048, 1]
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            # [1, 256, 2048, 1]
            Gc_joint = self.gg_channel(Gc_joint)
            # [1, 128, 2048, 1]
            g_xc = self.gx_channel(xc)
            # [1, 1, 2048, 1]
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            # [1, 257, 2048, 1]
            yc = torch.cat((g_xc, Gc_joint), 1)
            # [1, 2048, 1, 1]
            W_yc = self.W_channel(yc).transpose(1, 2)
            # [1, 2048, 32, 32]
            out = F.sigmoid(W_yc) * x

            return out

