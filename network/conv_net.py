import math

import torch
from torch.nn import functional as F, init
import torch.nn as nn


class WeightNormConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        # self.weight will be our v
        self.weight_g = torch.nn.Parameter(torch.Tensor(out_channels, 1, 1, 1))
        self.init_weight_g()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mobn = MeanOnlyBatchNorm(out_channels, affine=False, track_running_stats=True, **factory_kwargs)

    
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
        """
        The function performs a forward pass through a convolutional layer with normalized weights.
        
        :param input: The `input` parameter in the `forward` method represents the input data or image
        that will be passed through the convolutional layer during the forward pass of the neural
        network. It is typically a tensor representing the input image or feature map
        :return: The `forward` method is returning the result of applying a 2D convolution operation
        (`F.conv2d`) on the input tensor using the adjusted weight tensor, bias, stride, padding,
        dilation, and groups specified in the method parameters.
        """
        
        # add the mean only batch norm probly don't work well for inference but anyway
        # weight = self.weight * (self.weight_g / torch.linalg.vector_norm(self.weight, dim=(1, 2, 3), keepdim=True))
        # unnormalized_output =  F.conv2d(input, weight, None, self.stride, self.padding, self.dilation, self.groups)
        
        # normed_output = self.mobn(unnormalized_output)
        
        # return normed_output + self.bias.view([1, self.out_channels, 1, 1])
        
        # mean = input.mean(dim=[0, 2, 3], keepdim=True)
        
        # return input - mean + self.bias.view([1, self.out_channels, 1, 1])
        
        weight = self.weight * (self.weight_g / torch.linalg.vector_norm(self.weight, dim=(1, 2, 3), keepdim=True))
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
        
    
class WeightNormLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        # self.weight will be our v
        self.weight_g = torch.nn.Parameter(torch.Tensor(out_features, 1))
        
        # Mean Only Batch Norm (integrated)
        # Create MeanOnlyBatchNorm on the same device and dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mobn = MeanOnlyBatchNorm(out_features, affine=False, track_running_stats=True, **factory_kwargs)
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
        
        
        # weight = self.weight * (self.weight_g / torch.linalg.vector_norm(self.weight, dim=(1), keepdim=True))
        
        
        # add the mean only batch norm probly don't work well for inference but anyway
        # unnormalized_output = F.linear(input, self.weight)
        # # mean = torch.mean(input, dim=1, keepdim=True)
        
        # # input = input - mean
        # normed_output = self.mobn(unnormalized_output)
        
        
        # scaler = self.weight_g / torch.linalg.vector_norm(self.weight, dim=(1), keepdim=True)
        # return normed_output * scaler.view([1, self.out_features]) + self.bias.view([1, self.out_features])
        
        weight = self.weight * (self.weight_g / torch.linalg.vector_norm(self.weight, dim=(1), keepdim=True))
        return F.linear(input, weight, self.bias)
    
class MeanOnlyBatchNorm(nn.Module):
    """
    Base class for Mean-Only Batch Normalization.
    Only subtracts the mean, does not divide by standard deviation.
    Tracks running mean and includes optional affine parameters (gamma, beta).
    """
    def __init__(self, num_features, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features, **factory_kwargs)) # gamma
            self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))   # beta
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            # No running_var needed
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=device))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight) # Initialize gamma to 1
            nn.init.zeros_(self.bias)  # Initialize beta to 0

    # def _check_input_dim(self, input):
    #     raise NotImplementedError

    def forward(self, input):
        # self._check_input_dim(input)

        # Determine the dimensions to calculate the mean over
        # For N, C, H, W -> dims = (0, 2, 3)
        # For N, C, L -> dims = (0, 2)
        # For N, C -> dims = (0)
        reduce_dims = [0] + list(range(2, input.dim()))

        # Reshape shape for broadcasting (e.g., (1, C, 1, 1) for 2D)
        view_shape = [1, self.num_features] + [1] * (input.dim() - 2)

        if self.training and self.track_running_stats:
            # Calculate batch mean
            batch_mean = torch.mean(input, dim=reduce_dims, keepdim=False) # Shape (C,)

            # Update running mean
            # Using PyTorch's internal way slightly adjusted
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                # Use exponential moving average
                # new_running = (1 - momentum) * current_running + momentum * batch_stat
                # Note: PyTorch BN uses slightly different momentum update
                # running_mean = running_mean * (1 - momentum) + batch_mean * momentum
                # Let's stick to the common definition for clarity here:
                # EMA: X_t = (1-alpha)*X_{t-1} + alpha*current_val
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()

            mean_to_use = batch_mean.view(view_shape) # Reshape for broadcasting

        elif not self.training and self.track_running_stats and self.running_mean is not None:
            # Use running mean during evaluation
            mean_to_use = self.running_mean.view(view_shape)
        else:
            # Use batch mean if not training but not tracking stats,
            # or if training and not tracking stats
             mean_to_use = torch.mean(input, dim=reduce_dims, keepdim=True) # Keep dim here


        # --- Mean Normalization ---
        # Subtract the mean
        out = input - mean_to_use
        # --------------------------

        # Apply affine transformation (gamma * x_norm + beta)
        if self.affine:
            out = out * self.weight.view(view_shape) + self.bias.view(view_shape)

        return out

    def extra_repr(self):
        return '{num_features}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)


class MTConvNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv_1 = WeightNormConv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding="same")
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
        
        
        
