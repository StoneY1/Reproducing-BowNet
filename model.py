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

class ResidualBlock(nn.Module):
    """Basic 2-layer residual block as described in the original ResNet paper https://arxiv.org/abs/1512.03385"""
    def __init__(self, in_channels, out_channels, kernel_size, downsample_factor):
        super().__init__()
        self.c_in = in_channels
        self.c_out = out_channels
        self.ksize = kernel_size
        self.ds_factor = downsample_factor
        self.dims_match = self.c_out == self.c_in and self.ds_factor == 1
        self.conv1 = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=self.ksize, stride=self.ds_factor, padding=self.ksize//2)
        self.bn1 = nn.BatchNorm2d(num_features=self.c_out)
        self.conv2 = nn.Conv2d(in_channels=self.c_out, out_channels=self.c_out, kernel_size=self.ksize, stride=1, padding=self.ksize//2)
        self.bn2 = nn.BatchNorm2d(num_features=self.c_out)
        if not self.dims_match:
            self.dims_projecting_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=1, stride=self.ds_factor)
    
    def forward(self, input_tensor):
        """Forward pass of this block"""
        x = self.bn1(self.conv1(input_tensor))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        residual = input_tensor if self.dims_match else self.dims_projecting_conv(input_tensor)
        x = F.relu(residual + x)
        return x

class BowNet(nn.Module):
    """Class definition for our BowNet CNN. This arch is used for both the RotNet pretraining and the BowNet encoder.
    We use a straightforward ResNet18-like architecture with some slight modifications for CIFAR-100's 32x32 input resolution."""
    def __init__(self, num_classes):
        """Define layers"""
        super().__init__()
        self.num_classes = num_classes
        self.conv1_64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_64 = nn.BatchNorm2d(num_features=64)

        self.resblock1_64a = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=2)
        self.resblock1_64b = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=1)
        self.resblock2_128a = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock2_128b = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock3_256a = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, downsample_factor=2)
        self.resblock3_256b = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1)
        self.resblock4_512a = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, downsample_factor=1)
        self.resblock4_512b = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, downsample_factor=1)
        
        self.global_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc_out = nn.Linear(512, self.num_classes)

    def forward(self, input_tensor):
        """Forward pass of our BowNet-lite"""

        x = F.relu(self.bn1_64(self.conv1_64(input_tensor)))
        x = self.resblock1_64a(x)
        x = self.resblock1_64b(x)
        x = self.resblock2_128a(x)
        x = self.resblock2_128b(x)
        x = self.resblock3_256a(x)
        x = self.resblock3_256b(x)
        
        # We will need these feature maps for the K-means clustering to create a Visual BoW vocabulary
        self.resblock3_256b_fmaps = x 

        x = self.resblock4_512a(x)
        x = self.resblock4_512b(x)
        
        x = self.global_avg_pool(x).reshape(-1, 1, 512)
        x = self.fc_out(x)
        logits = x
        preds = F.softmax(x, dim=-1)

        return logits, preds

if __name__ == "__main__":
    bownet = BowNet(num_classes=100)
    #test_tensor = torch.transpose(torch.randn((50000, 32, 32, 3)), 1, 3)
    test_tensor = torch.transpose(torch.randn((5, 32, 32, 3)), 1, 3)

    test_logits, test_preds = bownet(test_tensor)


