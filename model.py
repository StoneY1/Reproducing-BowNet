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
from layers import ResidualBlock, NormalizedLinear

class BowNet(nn.Module):
    """Class definition for our BowNet CNN. This arch is used for both the RotNet pretraining and the BowNet encoder.
    We use a straightforward ResNet18-like architecture with some slight modifications for CIFAR-100's 32x32 input resolution."""
    def __init__(self, num_classes, bow_training=False):
        """Define layers"""
        super().__init__()
        print("[**] Using BowNet1")
        self.num_classes = num_classes
        self.bow_training = bow_training
        self.conv1_64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_64 = nn.BatchNorm2d(num_features=64)
        self.relu1_64 = nn.ReLU()

        # Resblock 1
        self.resblock1_64a = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=2)
        self.resblock1_64b = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3)

        # Resblock 2
        self.resblock2_128a = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3)
        self.resblock2_128b = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3)

        # Resblock 3
        self.resblock3_256a = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, downsample_factor=2)
        self.resblock3_256b = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3)

        # Rotation prediction head
        self.resblock4_512a = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3)
        self.resblock4_512b = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3)

        self.global_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        if self.bow_training:
            self.fc_fin = 256
            self.fc_out = NormalizedLinear(self.fc_fin, self.num_classes)
        else:
            self.fc_out = nn.Linear(512, self.num_classes)
            self.fc_fin = 512
        self.initialize()

    def forward(self, input_tensor):
        """Forward pass of BowNet-lite"""

        x = self.relu1_64(self.bn1_64(self.conv1_64(input_tensor)))
        x = self.resblock1_64a(x)
        x = self.resblock1_64b(x)

        self.resblock1_64_fmaps = x

        x = self.resblock2_128a(x)
        x = self.resblock2_128b(x)
        
        # We experimented with using different feature maps for training the linear classifier on CIFAR-100 classification
        self.resblock2_128_fmaps = x

        x = self.resblock3_256a(x)
        x = self.resblock3_256b(x)

        # We will need these feature maps for the K-means clustering to create a Visual BoW vocabulary
        self.resblock3_256_fmaps = x

        if not self.bow_training:
            # Only add these residual layers for the Rotation pre-training
            x = self.resblock4_512a(x)
            x = self.resblock4_512b(x)

        x = self.global_avg_pool(x).reshape(-1, self.fc_fin)
        x = self.fc_out(x)
        logits = x
        # print("bownet x shape", x.shape)
        preds = F.softmax(x, dim=-1)

        return logits, preds

    def initialize(self):
        for m in self._modules:
            block = self._modules[m]
            if not isinstance(block, (nn.Linear, nn.Conv2d)):
                continue
            fin = block.in_features if isinstance(block, nn.Linear) else block.in_channels * np.prod(block.kernel_size)
            fout = block.out_features if isinstance(block, nn.Linear) else block.out_channels * np.prod(block.kernel_size)
            limit = np.sqrt(6.0 / (fout + fin))
            block.weight.data.uniform_(-limit, limit)
            if block.bias is not None:
                block.bias.data.fill_(0.0)

class BowNet2(nn.Module):
    """
    CNN that is more like WRN-24-4
    """
    def __init__(self, num_classes, bow_training=False):
        """Define layers"""
        super().__init__()
        print("[**] Using BowNet2")
        self.num_classes = num_classes
        self.bow_training = bow_training
        self.conv1_64 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1_64 = nn.BatchNorm2d(num_features=16)
        self.relu1_64 = nn.ReLU()

        # ResBlock 1
        self.resblock1_64a = ResidualBlock(in_channels=16, out_channels=64, kernel_size=3)
        self.resblock1_64b = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3)
        self.resblock1_64c = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3)
        self.resblock1_64d = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3)

        # Resblock 2
        self.resblock2_128a = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, downsample_factor=2)
        self.resblock2_128b = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3)
        self.resblock2_128c = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3)
        self.resblock2_128d = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3)
            
        # Resblock 3
        self.resblock3_256a = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, downsample_factor=2)
        self.resblock3_256b = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3)
        self.resblock3_256c = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3)
        self.resblock3_256d = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3)

        # Extra prediction head that is used for RotNet
        self.resblock4_512a = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, use_dropout=True)
        self.resblock4_512b = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, use_dropout=True)
        self.resblock4_512c = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, use_dropout=True)
        self.resblock4_512d = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, use_dropout=True)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        if self.bow_training:
            self.fc_fin = 256
            self.fc_out = NormalizedLinear(self.fc_fin, self.num_classes)
        else:
            self.fc_fin = 512
            self.fc_out = nn.Linear(self.fc_fin, self.num_classes)

        self.initialize()

    def forward(self, input_tensor):

        x = self.relu1_64(self.bn1_64(self.conv1_64(input_tensor)))
        x = self.resblock1_64a(x)
        x = self.resblock1_64b(x)
        x = self.resblock1_64c(x)
        x = self.resblock1_64d(x)
        self.resblock1_64_fmaps = x

        x = self.resblock2_128a(x)
        x = self.resblock2_128b(x)
        x = self.resblock2_128c(x)
        x = self.resblock2_128d(x)
        
        # We experimented with using different feature maps for training the linear classifier on CIFAR-100 classification
        self.resblock2_128_fmaps = x

        x = self.resblock3_256a(x)
        x = self.resblock3_256b(x)
        x = self.resblock3_256c(x)
        x = self.resblock3_256d(x)

        # We will need these feature maps for the K-means clustering to create a Visual BoW vocabulary
        self.resblock3_256_fmaps = x

        if not self.bow_training:
            # Only add these residual layers for the Rotation pre-training
            x = self.resblock4_512a(x)
            x = self.resblock4_512b(x)
            x = self.resblock4_512c(x)
            x = self.resblock4_512d(x)

        x = self.global_avg_pool(x).reshape(-1, self.fc_fin)
        x = self.fc_out(x)
        logits = x
        # print("bownet x shape", x.shape)
        preds = F.softmax(x, dim=-1)

        return logits, preds

    def initialize(self):
        for m in self._modules:
            block = self._modules[m]
            if not isinstance(block, (nn.Linear, nn.Conv2d)):
                continue
            fin = block.in_features if isinstance(block, nn.Linear) else block.in_channels * np.prod(block.kernel_size)
            fout = block.out_features if isinstance(block, nn.Linear) else block.out_channels * np.prod(block.kernel_size)
            limit = np.sqrt(6.0 / (fout + fin))
            block.weight.data.uniform_(-limit, limit)
            if block.bias is not None:
                block.bias.data.fill_(0.0)

class WRN_28_K(nn.Module):
    """
    Wide residual network with 28 Conv layers and programmable multiplier K
    """
    def __init__(self, num_classes, K=10, bow_training=False):
        """Define layers"""
        super().__init__()
        print(f"[**] Using WRN-28-{K}")
        self.num_classes = num_classes
        self.K = K
        self.bow_training = bow_training
        self.conv1_64 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1_64 = nn.BatchNorm2d(num_features=16)
        self.relu1_64 = nn.ReLU()

        # ResBlock 1
        self.resblock1a = ResidualBlock(in_channels=16, out_channels=16*self.K, kernel_size=3, use_dropout=True)
        self.resblock1b = ResidualBlock(in_channels=16*self.K, out_channels=16*self.K, kernel_size=3, use_dropout=True)
        self.resblock1c = ResidualBlock(in_channels=16*self.K, out_channels=16*self.K, kernel_size=3, use_dropout=True)
        self.resblock1d = ResidualBlock(in_channels=16*self.K, out_channels=16*self.K, kernel_size=3, use_dropout=True)

        # Resblock 2
        self.resblock2a = ResidualBlock(in_channels=16*self.K, out_channels=32*self.K, kernel_size=3, downsample_factor=2, use_dropout=True)
        self.resblock2b = ResidualBlock(in_channels=32*self.K, out_channels=32*self.K, kernel_size=3, use_dropout=True)
        self.resblock2c = ResidualBlock(in_channels=32*self.K, out_channels=32*self.K, kernel_size=3, use_dropout=True)
        self.resblock2d = ResidualBlock(in_channels=32*self.K, out_channels=32*self.K, kernel_size=3, use_dropout=True)
            
        # Resblock 3
        self.resblock3a = ResidualBlock(in_channels=32*self.K, out_channels=64*self.K, kernel_size=3, downsample_factor=2, use_dropout=True)
        self.resblock3b = ResidualBlock(in_channels=64*self.K, out_channels=64*self.K, kernel_size=3, use_dropout=True)
        self.resblock3c = ResidualBlock(in_channels=64*self.K, out_channels=64*self.K, kernel_size=3, use_dropout=True)
        self.resblock3d = ResidualBlock(in_channels=64*self.K, out_channels=64*self.K, kernel_size=3, use_dropout=True)

        # Extra prediction head that is used for RotNet
        self.resblock4a = ResidualBlock(in_channels=64*self.K, out_channels=64*self.K, kernel_size=3, use_dropout=True)
        self.resblock4b = ResidualBlock(in_channels=64*self.K, out_channels=64*self.K, kernel_size=3, use_dropout=True)
        self.resblock4c = ResidualBlock(in_channels=64*self.K, out_channels=64*self.K, kernel_size=3, use_dropout=True)
        self.resblock4d = ResidualBlock(in_channels=64*self.K, out_channels=64*self.K, kernel_size=3, use_dropout=True)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.flatten = nn.Flatten()
        if self.bow_training:
            self.fc_fin = 64*self.K
            self.fc_out = NormalizedLinear(self.fc_fin, self.num_classes)
        else:
            self.fc_fin = 64*self.K
            self.fc_out = nn.Linear(self.fc_fin, self.num_classes)

        self.initialize()

    def forward(self, input_tensor):

        x = self.relu1_64(self.bn1_64(self.conv1_64(input_tensor)))
        x = self.resblock1a(x)
        x = self.resblock1b(x)
        x = self.resblock1c(x)
        x = self.resblock1d(x)
        self.resblock1_fmaps = x

        x = self.resblock2a(x)
        x = self.resblock2b(x)
        x = self.resblock2c(x)
        x = self.resblock2d(x)
        
        # We experimented with using different feature maps for training the linear classifier on CIFAR-100 classification
        self.resblock2_fmaps = x

        x = self.resblock3a(x)
        x = self.resblock3b(x)
        x = self.resblock3c(x)
        x = self.resblock3d(x)

        # We will need these feature maps for the K-means clustering to create a Visual BoW vocabulary
        self.resblock3_fmaps = x

        if not self.bow_training:
            # Only add these residual layers for the Rotation pre-training
            x = self.resblock4a(x)
            x = self.resblock4b(x)
            x = self.resblock4c(x)
            x = self.resblock4d(x)

            x = self.global_avg_pool(x).reshape(-1, self.fc_fin)
        else:
            x = self.global_avg_pool(x).reshape(-1, self.fc_fin)
            #x = self.flatten(x)

        x = self.fc_out(x)
        logits = x
        preds = F.softmax(x, dim=-1)

        return logits, preds

    def initialize(self):
        for m in self._modules:
            block = self._modules[m]
            if not isinstance(block, (nn.Linear, nn.Conv2d)):
                continue
            fin = block.in_features if isinstance(block, nn.Linear) else block.in_channels * np.prod(block.kernel_size)
            fout = block.out_features if isinstance(block, nn.Linear) else block.out_channels * np.prod(block.kernel_size)
            limit = np.sqrt(6.0 / (fout + fin))
            block.weight.data.uniform_(-limit, limit)
            if block.bias is not None:
                block.bias.data.fill_(0.0)

class LinearClassifier(nn.Module):

    def __init__(self, num_classes, num_channels_in=256, spatial_size=8):
        """Define layers"""
        super().__init__()
        self.num_classes = num_classes
        self.num_channels_in = num_channels_in
        self.spatial_size = spatial_size
        self.global_avg_pool = nn.AvgPool2d(kernel_size=self.spatial_size, stride=1)
        self.adaptive_max_pool = nn.AdaptiveAvgPool2d((8,8))
        self.batch_norm = nn.BatchNorm2d(self.num_channels_in)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.num_channels_in*self.spatial_size*self.spatial_size,self.num_classes)

        self.initialize()

    def forward(self, input_tensor):

        x = input_tensor

        x = self.flatten(x)

        x = self.fc1(x)
        x = x

        logits = x
        preds = F.softmax(x, dim=-1)

        return logits, preds

    def initialize(self):
        for m in self._modules:
            block = self._modules[m]
            if isinstance(block, nn.Linear):
                fin = block.in_features
                fout = block.out_features
                std_val = np.sqrt(2.0 / fout)
                block.weight.data.normal_(0.0, std_val)
                if block.bias is not None:
                    block.bias.data.fill_(0.0)

class NonLinearClassifier(nn.Module):

    def __init__(self, num_classes, num_channels_in=256, spatial_size=8):
        """Define layers"""
        super().__init__()
        self.num_classes = num_classes
        self.num_channels_in = num_channels_in
        self.spatial_size = spatial_size
        self.input_vector_len = self.num_channels_in*self.spatial_size*self.spatial_size

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_vector_len, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(512, self.num_classes)

        self.initialize()

    def forward(self, input_tensor):

        x = self.flatten(input_tensor)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc_out(x)

        x = x
        logits = x
        preds = F.softmax(x, dim=-1)

        return logits, preds

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                limit = np.sqrt(6.0 / (fout + fin))
                m.weight.data.uniform_(-limit, limit)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)



if __name__ == "__main__":
    bownet = BowNet(num_classes=100, bow_training=True)
    #linear = LinearClassifier(num_classes=100)
    #test_tensor = torch.transpose(torch.randn((50000, 32, 32, 3)), 1, 3)
    test_tensor = torch.transpose(torch.randn((2, 32, 32, 3)), 1, 3)
    #test_tensor = torch.randn((1, 256, 8, 8))

    test_logits, test_preds = bownet(test_tensor)
    #test_logits, test_preds = linear(test_tensor)
