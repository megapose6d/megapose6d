"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Third Party
import torch
import torch.nn.functional as F
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlockV2(nn.Module):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): stride size.
        downsample (Module) optional downsample module to downsample the input.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        residual = self.downsample(out) if self.downsample is not None else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        return out + residual


class WideResNet(nn.Module):
    def __init__(self, block, layers, width, num_inputs=3, maxpool=True):
        super().__init__()

        config = [int(v * width) for v in (64, 128, 256, 512)]
        self.inplanes = config[0]
        self.conv1 = nn.Conv2d(
            num_inputs, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(block, config[0], layers[0])
        self.layer2 = self._make_layer(block, config[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, config[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, config[3], layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(
                self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False
            )

        layers = [
            block(self.inplanes, planes, stride, downsample),
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


CONFIG = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}


class WideResNet18(WideResNet):
    def __init__(self, n_inputs=3, width=1.0):
        super().__init__(block=BasicBlockV2, layers=CONFIG[18], width=width, num_inputs=n_inputs)
        self.n_features = int(512 * width)


class WideResNet34(WideResNet):
    def __init__(self, n_inputs=3, width=1.0):
        super().__init__(block=BasicBlockV2, layers=CONFIG[34], width=width, num_inputs=n_inputs)
        self.n_features = int(512 * width)


if __name__ == "__main__":
    model = WideResNet34(n_inputs=3, width=4)
    x = torch.randn(1, 3, 224, 224)
    outputs = model(x)
    print(outputs.shape)
