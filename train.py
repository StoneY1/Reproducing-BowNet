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
from model import BowNet
from load_data_test import dloader_train, dloader_test

# pip install kmeans-pytorch
from kmeans_pytorch import kmeans
# alternatively, we can use SKLearn
from sklearn.cluster import KMeans


def train_rotation():
    """After setting up optimizer and dataset, this function is the main training loop for rotation classification"""
    # Just writing pseudo-code for now
    num_epochs = 200
    bownet = BowNet(num_classes=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(bownet.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dloader_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, preds = bownet(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return

def build_RotNet_vocab(bownet: BowNet, K: int=2048):
    """After training RotNet, we will get the bag-of-words vocabulary using Euclidean distance based KMeans clustering"""
    feature_vectors_list = []

    # Iterate through trainset, compiling set of feature maps before performing KMeans clustering
    for i, data in enumerate(dloader_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data

            # Forward data through bownet, then get resblock_3_256b fmaps
            # The authors propose densely sampling feature C-dimensional feature vectors at each
            # spatial location where C is the size of the channel dimension. These feature vectors are used for the KMeans clustering
            outputs = bownet(inputs)
            # TODO need to verify that bownet.resblock3_256b_fmaps is actually getting updated after calling bownet.forward()
            resblock3_fmaps = bownet.resblock3_256b_fmaps
            resblock3_feature_vectors = resblock3_fmaps.reshape(-1, resblock3_fmaps.shape[1])
            feature_vectors_list.extend(resblock3_feature_vectors)

    rotnet_feature_vectors = np.array(feature_vectors_list)

    '''
    # kmeans using pytorch; this package doesn't seem to have a "predict" method
    cluster_ids_x, cluster_centers = kmeans(
        X=rotnet_feature_vectors, num_clusters=K, distance='euclidean', device=torch.device('cuda:0')
    )
    '''

    # Kmeans using sklearn
    sk_kmeans = KMeans(n_clusters=K).fit(rotnet_feature_vectors)
    rotnet_vocab = sk_kmeans.cluster_centers_

    return sk_kmeans, rotnet_vocab

# TODO Need to implement the histogram creation. maybe
