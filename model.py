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
        self.num_classes = num_classes
        self.conv1_64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_64 = nn.BatchNorm2d(num_features=64)
        self.relu1_64 = nn.ReLU()

        self.resblock1_64a = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=2)
        self.resblock1_64b = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=1)
        self.resblock2_128a = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock2_128b = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock3_256a = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, downsample_factor=2, use_dropout=False, dropout_rate=0.3)
        self.resblock3_256b = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1, use_dropout=False, dropout_rate=0.3)
        self.resblock4_512a = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, downsample_factor=1, use_dropout=True, dropout_rate=0.5)
        self.resblock4_512b = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, downsample_factor=1, use_dropout=True, dropout_rate=0.5)

        self.global_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        if bow_training:
            self.fc_out = NormalizedLinear(512, self.num_classes)
        else:
            self.fc_out = nn.Linear(512, self.num_classes)
        # self.fc_out = nn.Linear(512,1)
        self.initialize()

    def forward(self, input_tensor):
        """Forward pass of our BowNet-lite"""

        x = self.relu1_64(self.bn1_64(self.conv1_64(input_tensor)))
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
    """Class definition for our BowNet CNN. This arch is used for both the RotNet pretraining and the BowNet encoder.
    We use a straightforward ResNet18-like architecture with some slight modifications for CIFAR-100's 32x32 input resolution."""
    def __init__(self, num_classes, bow_training=False):
        """Define layers"""
        super().__init__()
        self.num_classes = num_classes
        self.conv1_64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn1_64 = nn.BatchNorm2d(num_features=64)
        self.relu1_64 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2,2))

        self.resblock1_64a = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=1)
        self.resblock1_64b = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=1)
        self.resblock2_128a = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock2_128b = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock2_128c = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock3_256a = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, downsample_factor=2)
        self.resblock3_256b = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1, use_dropout=True)
        self.resblock3_256c = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1, use_dropout=True)
        self.resblock4_512a = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, downsample_factor=1, use_dropout=True)
        self.resblock4_512b = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, downsample_factor=1, use_dropout=True)
        self.resblock4_512c = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, downsample_factor=1, use_dropout=True)
        self.resblock4_512d = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, downsample_factor=1, use_dropout=True)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        if bow_training:
            self.fc_out = NormalizedLinear(512, self.num_classes)
        else:
            self.fc_out = nn.Linear(512, self.num_classes)
        self.initialize()

    def forward(self, input_tensor):
        """Forward pass of our BowNet-lite"""

        x = self.relu1_64(self.bn1_64(self.conv1_64(input_tensor)))
        x = self.maxpool_1(x)
        x = self.resblock1_64a(x)
        x = self.resblock1_64b(x)
        self.resblock1_64b_fmaps = x
        
        x = self.resblock2_128a(x)
        self.resblock2_128a_fmaps = x
        x = self.resblock2_128b(x)
        self.resblock2_128b_fmaps = x
        x = self.resblock2_128c(x)
        self.resblock2_128c_fmaps = x

        x = self.resblock3_256a(x)
        x = self.resblock3_256b(x)
        x = self.resblock3_256c(x)

        # We will need these feature maps for the K-means clustering to create a Visual BoW vocabulary
        self.resblock3_256b_fmaps = x

        x = self.resblock4_512a(x)
        x = self.resblock4_512b(x)
        x = self.resblock4_512c(x)
        x = self.resblock4_512d(x)

        x = self.global_avg_pool(x).reshape(-1, 1, 512)
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

class BowNet3(nn.Module):
    """Class definition for our BowNet CNN. This arch is used for both the RotNet pretraining and the BowNet encoder.
    We use a straightforward ResNet18-like architecture with some slight modifications for CIFAR-100's 32x32 input resolution."""
    def __init__(self, num_classes, bow_training=False):
        """Define layers"""
        super().__init__()
        self.num_classes = num_classes
        self.conv1_64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn1_64 = nn.BatchNorm2d(num_features=64)
        self.relu1_64 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2,2))

        self.resblock1_64a = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=1)
        self.resblock1_64b = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=1)
        self.resblock1_64c = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, downsample_factor=1)
        self.resblock2_128a = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock2_128b = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock2_128c = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock2_128d = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, downsample_factor=1)
        self.resblock3_256a = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, downsample_factor=2)
        self.resblock3_256b = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1)
        self.resblock3_256c = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1)
        self.resblock3_256d = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1)
        self.resblock3_256e = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1)
        self.resblock3_256f = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, downsample_factor=1)
        self.resblock4_512a = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, downsample_factor=1)
        self.resblock4_512b = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, downsample_factor=1)
        self.resblock4_512c = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, downsample_factor=1)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        if bow_training:
            self.fc_out = NormalizedLinear(512, self.num_classes)
        else:
            self.fc_out = nn.Linear(512, self.num_classes)
        self.initialize()

    def forward(self, input_tensor):
        """Forward pass of our BowNet-lite"""

        x = self.relu1_64(self.bn1_64(self.conv1_64(input_tensor)))
        x = self.maxpool_1(x)
        x = self.resblock1_64a(x)
        x = self.resblock1_64b(x)
        x = self.resblock1_64c(x)
        self.resblock1_64c_fmaps = x

        x = self.resblock2_128a(x)
        x = self.resblock2_128b(x)
        x = self.resblock2_128c(x)
        x = self.resblock2_128d(x)
        self.resblock2_128d_fmaps = x

        x = self.resblock3_256a(x)
        x = self.resblock3_256b(x)
        x = self.resblock3_256c(x)
        x = self.resblock3_256d(x)
        x = self.resblock3_256e(x)
        x = self.resblock3_256f(x)

        # We will need these feature maps for the K-means clustering to create a Visual BoW vocabulary
        self.resblock3_256f_fmaps = x

        x = self.resblock4_512a(x)
        x = self.resblock4_512b(x)
        x = self.resblock4_512c(x)

        x = self.global_avg_pool(x).reshape(-1, 1, 512)
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
        #self.fc1 = nn.Linear(6400,self.num_classes)
        self.fc2 = nn.Linear(1000,self.num_classes)

        self.initilize()


    def forward(self, input_tensor):
        # x = input_tensor.view(input_tensor.size(0), -1)
        # x = self.global_avg_pool(input_tensor)
        #x = self.adaptive_max_pool(input_tensor)
        # print("adaptive",x.shape)

        #x = self.batch_norm(x)

        # print("batch_norm ",x.shape)
        x = input_tensor

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

        x = x.reshape(-1, 1, self.num_classes)
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


def load_checkpoint(checkpoint,device, bownet_arch):
    bownet = bownet_arch(num_classes=4).to(device)
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
