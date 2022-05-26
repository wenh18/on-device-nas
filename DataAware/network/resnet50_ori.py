
import torch
import torchvision.models.resnet as Resnet
import torch.nn as nn

def cifar100_resnet50_ori(pre_model=False):
    # This Module is based on Resnet-50 for dataset CIFAR100
    model = Resnet.ResNet(Resnet.Bottleneck,[3,4,6,3],num_classes=100)
    model.inplanes=64
    model.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    model.bn1 = nn.BatchNorm2d(64)
    model.linear = nn.Linear(512*Resnet.Bottleneck.expansion, 100)
    del model.maxpool
    model.maxpool = lambda x : x
    if pre_model:
        model.load_state_dict(torch.load('/root/resnet20/model/premodel/resnet50-200-best_78.0099.pth'))
    return model