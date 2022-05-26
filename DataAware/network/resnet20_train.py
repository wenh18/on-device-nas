'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from select import select
import sys
from tkinter import FALSE
from traceback import print_tb

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
import math

from matplotlib.pyplot import flag
import torch.nn as nn
import torch
import numpy as np
import global_var

global_var._init()
threshold = -1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# https://github.com/nibuiro/CondConv-pytorch/blob/master/condconv/condconv.py
class MOEconv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros', num_experts=10, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = kernel_size // 2 if isinstance(kernel_size, int) else kernel_size[0] // 2
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MOEconv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        # self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs, expert_weights):
        expert_weights = torch.tensor(expert_weights).cuda()
        # b, _, _, _ = inputs.size()
        # res = []
        # for input in inputs:
        #     input = input.unsqueeze(0)
        #     pooled_inputs = self._avg_pooling(input)
        #     routing_weights = self._routing_fn(pooled_inputs)
        kernels = torch.sum(expert_weights[:, None, None, None, None] * self.weight, 0)
        out = self._conv_forward(inputs, kernels)
            # res.append(out)
        return out  # torch.cat(res, dim=0)

class MOEClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    def __init__(self, num_expert: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MOEClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((num_expert, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(num_expert, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, expert_weights) -> Tensor:
        expert_weights = torch.tensor(expert_weights).cuda()
        weight = torch.sum(expert_weights[:, None, None] * self.weight, 0)
        if self.bias is not None:
            bias = torch.sum(expert_weights[:, None] * self.bias, 0)
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, weight, None)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, moeflag=False, num_experts=10):
        super(BasicBlock, self).__init__()
        self.moeflag = moeflag
        self.num_experts = num_experts
        if self.moeflag:
            self.conv1 = MOEconv(inplanes, planes, kernel_size=3, stride=stride,num_experts=num_experts)
            self.conv2 = MOEconv(planes, planes, kernel_size=3,num_experts=num_experts)
            self.conv1_pre = conv3x3(inplanes, planes, stride)
            self.conv2_pre = conv3x3(planes, planes)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        identity = x
        if self.moeflag:
            alpha = global_var.get_value('alpha')
            index = global_var.get_value('moe_index')
            if threshold > np.random.rand():
                out = self.conv1_pre(x)
            else:
                out = self.conv1(x,alpha[index])
            out = self.bn1(out)
            out = self.relu(out)
            index +=1
            if threshold > np.random.rand():
                out = self.conv2_pre(out)
            else:
                out = self.conv2(out,alpha[index])
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            global_var.set_value('moe_index', index+1)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, num_experts=10):
        super(CifarResNet, self).__init__()
        self.num_expert = num_experts
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.inplanes = self.next_inplanes
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, moeflag=True)
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)

        self.fc = MOEClassifier(num_expert=num_experts,in_features=64 * block.expansion,out_features=num_classes,bias=True)
        self.fc_pre = nn.Linear(in_features=64 * block.expansion,out_features=num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, moeflag=False):
        downsample = None
        bn = nn.BatchNorm2d(planes * block.expansion, track_running_stats=True)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                bn
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, moeflag=False))
        self.next_inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.next_inplanes, planes, moeflag=moeflag))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        alpha = global_var.get_value('alpha')
        index = global_var.get_value('moe_index')
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        if threshold > np.random.rand():
            out = self.fc_pre(x)
        else:
            out = self.fc(x,alpha[index])
        return out

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not module.weight.requires_grad:
                    module.eval()
        #         else:
        #             count += 1
        # print(count)

        # if count > 0:
        #     print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)


def cifar100_resnet20() -> CifarResNet:
    model = CifarResNet(BasicBlock, [3] * 3)
    return model
