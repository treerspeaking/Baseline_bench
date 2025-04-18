import torch

from network.conv_net import MTConvNet
from network.cifar_shakeshake26 import cifar_shakeshake26
from network.resnet50 import SingleNetwork

def net_factory(network, in_channels, num_classes, pretrained=None):
    if network == "ConvNet":
        return MTConvNet(in_channels, num_classes)
    if network == "cifar_shakeshake26":
        return cifar_shakeshake26(pretrained=False, num_classes=num_classes)
    if network == "resnet50_cps":
        return SingleNetwork(num_classes=num_classes, criterion=torch.nn.CrossEntropyLoss(), norm_layer=torch.nn.BatchNorm2d, pretrained=pretrained)
