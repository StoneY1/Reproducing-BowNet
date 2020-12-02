'''
2020 ML Reproducibility Challenge
Harry Nguyen, Stone Yun, Hisham Mohammad
Part of our submission for reproducing the CVPR 2020 paper: Learning Representations by Predicting Bags of Visual Words
https://arxiv.org/abs/2002.12247
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _Loss as Loss


class SoftCrossEntropyLoss(Loss):
    """Need to implement a cross entropy loss for soft-targets. 
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, output, target):
        log_preds = F.log_softmax(output, dim=-1)
        batchloss = -torch.sum(target * log_preds, dim=-1)
        if self.reduction == 'sum':
            soft_crossentropy_loss = batchloss.sum()
        elif self.reduction == 'mean':
                soft_crossentropy_loss = batchloss.mean()
        else:
            soft_crossentropy_loss = batchloss

        return soft_crossentropy_loss

class ResidualBlock(nn.Module):
    """Basic 2-layer residual block as described in the original ResNet paper https://arxiv.org/abs/1512.03385"""
    def __init__(self, in_channels, out_channels, kernel_size, downsample_factor, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        self.c_in = in_channels
        self.c_out = out_channels
        self.ksize = kernel_size
        self.ds_factor = downsample_factor
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.dims_match = self.c_out == self.c_in and self.ds_factor == 1

        self.conv1 = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=self.ksize, stride=self.ds_factor, padding=self.ksize//2)
        self.bn1 = nn.BatchNorm2d(num_features=self.c_out)
        self.relu1 = nn.ReLU()
        if self.use_dropout:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.conv2 = nn.Conv2d(in_channels=self.c_out, out_channels=self.c_out, kernel_size=self.ksize, stride=1, padding=self.ksize//2)
        self.bn2 = nn.BatchNorm2d(num_features=self.c_out)
        self.relu2 = nn.ReLU()
        self.relu_out = nn.ReLU()

        if not self.dims_match:
            self.dims_projecting_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=1, stride=self.ds_factor)

        self.initialize()

    def forward(self, input_tensor):
        """Forward pass of this block"""
        x = self.relu1(self.bn1(self.conv1(input_tensor)))
        
        if self.use_dropout:
            self.dropout(x)

        x = self.relu2(self.bn2(self.conv2(x)))

        residual = input_tensor if self.dims_match else self.dims_projecting_conv(input_tensor)
        x = self.relu_out(residual + x)
        return x
    
    def initialize(self):
        for m in self._modules:
            block = self._modules[m]
            if not isinstance(block, nn.Conv2d):
                continue
            fin = block.in_channels * np.prod(block.kernel_size)
            fout = block.out_channels * np.prod(block.kernel_size)
            limit = np.sqrt(6.0 / (fout + fin))
            block.weight.data.uniform_(-limit, limit)
            if block.bias is not None:
                block.bias.data.fill_(0.0)

class NormalizedLinear(nn.Module):
    """Normalized linear layer with scalar as described in paper.
    Behaves like a regular Linear/Fully Connected layer except we now need to multiply by a learnable scale
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.c_in = in_features
        self.c_out = out_features
        self.init_limit = np.sqrt(6/(self.c_in + self.c_out))
        
        # Initialize all relevant paramters. Weight tensor, normalized weight, gamma scalar
        self.weight = nn.Parameter(torch.Tensor(self.c_out, self.c_in))
        self.weight.data.uniform_(-self.init_limit, self.init_limit) # Classic GlorotUniform initialization
        self.normed_weight = nn.Parameter(self.weight/torch.norm(self.weight))
        self.gamma = nn.Parameter(torch.Tensor(1,1))
        self.gamma.data.fill_(1.0)

        # Based on equation in paper, this layer has no bias parameters
        self.register_parameter('bias', None)

    def forward(self, input_tensor):
        x = F.linear(input_tensor, self.normed_weight, self.bias)
        x *= self.gamma
        return x

