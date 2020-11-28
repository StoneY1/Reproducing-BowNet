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
        self.normed_weight = self.weight/torch.norm(self.weight)
        self.gamma = nn.Parameter(torch.Tensor(1,1))
        self.gamma.data.normal_()

        # Based on equation in paper, this layer has no bias parameters
        self.register_parameter('bias', None)

    def forward(self, input_tensor):
        x = F.linear(input_tensor, self.normed_weight, self.bias)
        x *= self.gamma
        return x

class BowNet(nn.Module):
    """Class definition for our BowNet CNN. This arch is used for both the RotNet pretraining and the BowNet encoder.
    We use a straightforward ResNet18-like architecture with some slight modifications for CIFAR-100's 32x32 input resolution."""
    def __init__(self, num_classes, bow_training=False):
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
        if bow_training:
            self.fc_out = NormalizedLinear(512, self.num_classes)
        else:
            self.fc_out = nn.Linear(512, self.num_classes)
        # self.fc_out = nn.Linear(512,1)

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
        # print("bownet x shape", x.shape)
        preds = F.softmax(x, dim=-1)
        #import pdb; pdb.set_trace()

        return logits, preds

class BowNet2(nn.Module):
    """Class definition for our BowNet CNN. This arch is used for both the RotNet pretraining and the BowNet encoder.
    We use a straightforward ResNet18-like architecture with some slight modifications for CIFAR-100's 32x32 input resolution."""
    def __init__(self, num_classes, bow_training=False):
        """Define layers"""
        super().__init__()
        self.num_classes = num_classes
        self.conv1_64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)
        self.bn1_64 = nn.BatchNorm2d(num_features=64)

        self.resblock1_64a = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=1)
        self.resblock1_64b = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=1)
        self.resblock2_128a = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, downsample_factor=2)
        self.resblock2_128b = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock3_256a = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, downsample_factor=2)
        self.resblock3_256b = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1)
        self.resblock4_512a = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, downsample_factor=1)
        self.resblock4_512b = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, downsample_factor=1)

        self.global_avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)
        if bow_training:
            self.fc_out = NormalizedLinear(512, self.num_classes)
        else:
            self.fc_out = nn.Linear(512, self.num_classes)
        # self.fc_out = nn.Linear(512,1)

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
        # print("bownet x shape", x.shape)
        preds = F.softmax(x, dim=-1)
        #import pdb; pdb.set_trace()

        return logits, preds




class LinearClassifier(nn.Module):

    def __init__(self, num_classes):
        """Define layers"""
        super().__init__()
        self.num_classes = num_classes
        self.global_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.adaptive_max_pool = nn.AdaptiveAvgPool2d((5,5))
        self.batch_norm = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6400,self.num_classes)
        self.fc2 = nn.Linear(1000,self.num_classes)

        self.initilize()


    def forward(self, input_tensor):
        # x = input_tensor.view(input_tensor.size(0), -1)
        # x = self.global_avg_pool(input_tensor)
        x = self.adaptive_max_pool(input_tensor)
        # print("adaptive",x.shape)

        x = self.batch_norm(x)

        # print("batch_norm ",x.shape)

        x = self.flatten(x)
        # print("flatten: ",x.shape)
        #
        # x= x.reshape(-1, 256)
        # x = input_tensor.view(-1, 16384)
        # print(x.shape)
        # x = F.relu(x)
        x = self.fc1(x)
        # print("fc1_out: ",x.shape)
        # x = F.relu(x)
        # # print(x.shape)
        # x = self.fc2(x)

        x = x.reshape(-1, 1, 100)

        logits = x
        preds = F.softmax(x, dim=-1)

        return logits, preds

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0 / fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

def load_checkpoint(checkpoint,device):
    bownet = BowNet(num_classes=4).to(device)
    optimizer = optim.SGD(bownet.parameters(), lr=0.1, momentum=0.9,weight_decay= 5e-4)

    bownet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return bownet, optimizer, epoch, loss

if __name__ == "__main__":
    bownet = BowNet(num_classes=100, bow_training=True)
    #linear = LinearClassifier(num_classes=100)
    #test_tensor = torch.transpose(torch.randn((50000, 32, 32, 3)), 1, 3)
    test_tensor = torch.transpose(torch.randn((2, 32, 32, 3)), 1, 3)
    #test_tensor = torch.randn((1, 256, 8, 8))

    test_logits, test_preds = bownet(test_tensor)
    #test_logits, test_preds = linear(test_tensor)
