import torch
import torch.nn as nn
import global_var

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, moeflag=False, num_experts=21):
        super(BasicBlock, self).__init__()
        self.moeflag = moeflag
        self.num_experts = num_experts
        if self.moeflag:
            self.conv1 = nn.ModuleList([conv3x3(inplanes, planes, stride) for _ in range(num_experts)])
            self.conv2 = nn.ModuleList([conv3x3(planes, planes) for _ in range(num_experts)])
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        alpha = global_var.get_value('alpha')
        if self.moeflag:
            out_tmp = self.conv1[0](x) * alpha[0]
            for i in range(1, self.num_experts):
                out_tmp += self.conv1[i](x) * alpha[i]
            out = out_tmp
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.moeflag:
            out_tmp = self.conv2[0](out) * alpha[0]
            for i in range(1, self.num_experts):
                out_tmp += self.conv2[i](out) * alpha[i]
            out = out_tmp
        else:
            out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, num_experts=21):
        super(CifarResNet, self).__init__()
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
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.ModuleList([nn.Linear(64 * block.expansion, num_classes) for _ in range(num_experts)])
        self.num_experts = num_experts
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
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, moeflag=moeflag))
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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        alpha = global_var.get_value('alpha')
        x_tmp = self.fc[0](x) * alpha[0]
        for i in range(1, self.num_experts):
            x_tmp += self.fc[i](x) * alpha[i]
        x = x_tmp
        return x

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not module.weight.requires_grad:
                    module.eval()
                    count += 1

        # if count > 0:
        #     print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)



def cifar100_resnet20() -> CifarResNet:
    model = CifarResNet(BasicBlock, [3] * 3)
    return model
