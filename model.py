import math

import torch
import torch.nn.functional as F
from torch.nn import functional as F, init


class WeightNormConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        # self.weight will be our v
        self.weight_g = torch.nn.Parameter(torch.Tensor(out_channels, 1, 1, 1))
        self.init_weight_g()
    
    def init_weight_g(self):
        
        # due to weight already has been init in the Conv2D
        # due to the reset_parameter function that has been called in the parent function
        # rename this to init_weight_g
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
        torch.nn.init.constant_(self.weight_g, 1.0)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)
            
    def forward(self, input):
        weight = self.weight * (self.weight_g / torch.linalg.vector_norm(self.weight, dim=(1, 2, 3), keepdim=True))
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class WeightNormLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        # self.weight will be our v
        self.weight_g = torch.nn.Parameter(torch.Tensor(out_features, 1))
        self.init_weight_g()
        
    def init_weight_g(self):
        # due to weight already has been init in the Conv2D
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        torch.nn.init.constant_(self.weight_g, 1.0)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)
    
    def forward(self, input):
        weight = self.weight * (self.weight_g / torch.linalg.vector_norm(self.weight, dim=(1), keepdim=True))
        return F.linear(input, weight, self.bias)

class MTConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv_1 = WeightNormConv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_features=128)
        self.lrelu_1 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.conv_2= WeightNormConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_features=128)
        self.lrelu_2 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.conv_3 = WeightNormConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.batch_norm_3 = torch.nn.BatchNorm2d(num_features=128)
        self.lrelu_3 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_1 = torch.nn.Dropout2d(p=0.5)
        
        self.conv_4 = WeightNormConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.batch_norm_4 = torch.nn.BatchNorm2d(num_features=256)
        self.lrelu_4 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.conv_5 = WeightNormConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.batch_norm_5 = torch.nn.BatchNorm2d(num_features=256)
        self.lrelu_5 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.conv_6 = WeightNormConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.batch_norm_6 = torch.nn.BatchNorm2d(num_features=256)
        self.lrelu_6 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_2 = torch.nn.Dropout2d(p=0.5)
        
        self.conv_7 = WeightNormConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="valid")
        self.batch_norm_7 = torch.nn.BatchNorm2d(num_features=512)
        self.lrelu_7 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.conv_8 = WeightNormConv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding="same")
        self.batch_norm_8 = torch.nn.BatchNorm2d(num_features=256)
        self.lrelu_8 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.conv_9 = WeightNormConv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding="same")
        self.batch_norm_9 = torch.nn.BatchNorm2d(num_features=128)
        self.lrelu_9 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.gap = torch.nn.AvgPool2d(kernel_size=6, stride=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = WeightNormLinear(128, num_classes)
        # self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.lrelu_1(x)
        
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.lrelu_2(x)
        
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.lrelu_3(x)
        x = self.max_pool_1(x)
        x = self.dropout_1(x)
        
        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.lrelu_4 (x)
        
        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.lrelu_5(x)
        
        x = self.conv_6(x)
        x = self. batch_norm_6(x)
        x = self.lrelu_6(x)
        x = self.max_pool_2(x)
        x = self.dropout_2(x)
        
        x = self.conv_7(x)
        x = self.batch_norm_7(x)
        x = self.lrelu_7(x)
        
        x = self.conv_8(x)
        x = self.batch_norm_8(x)
        x = self.lrelu_8(x)
        
        x = self.conv_9(x)
        x = self.batch_norm_9(x)
        x = self.lrelu_9(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.softmax(x)
        
        return x
        
        
        
        
        