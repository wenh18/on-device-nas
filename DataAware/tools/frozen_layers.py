def frozen_layers_resnet20(model):
    frozen_layers = [model.layer1, model.layer2]
    for layer in frozen_layers:
        for name, value in layer.named_parameters():
            value.requires_grad = False
    model.conv1.weight.requires_grad = False
    model.bn1.weight.requires_grad = False
    model.bn1.bias.requires_grad = False
    for name, value in model.named_parameters():
        if 'downsample' in name or 'bn' in name or '3.0' in name  or 'pre' in name:
            value.requires_grad = False
    # for name, value in model.layer3.named_parameters():
    #     if 'conv1.0.' in name or 'conv2.0.' in name:
    #         value.requires_grad = False
    return model


def frozen_layers_resnet50(model):

    frozen_layers = [model.layer1, model.layer2, model.layer3]
    for layer in frozen_layers:
        for name, value in layer.named_parameters():
            value.requires_grad = False

    model.conv1.weight.requires_grad = False
    model.bn1.weight.requires_grad = False
    model.bn1.bias.requires_grad = False

    for name, value in model.named_parameters():
        if 'layer4.0' in name:
            value.requires_grad = False
        if 'downsample' in name or 'bn' in name or 'fc.0.' in name:
            value.requires_grad = False
    for name, value in model.layer4.named_parameters():
        if 'conv1.0.' in name or 'conv2.0.' in name or 'conv3.0.' in name:
            value.requires_grad = False
    # for name, value in model.named_parameters():
    #     print(name,value.requires_grad)

    return model