from __future__ import print_function
import argparse
import os
import imp
from dataloader import DataLoader, GenericDataset
import matplotlib.pyplot as plt

from model import BowNet, load_checkpoint
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from kmeans_pytorch import kmeans

# Set train and test datasets and the corresponding data loaders
batch_size = 128

data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = False #Set to False then the label is the class - Set to True to get rotation label
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'cifar100'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = False #Set to False then the label is the class - Set to True to get rotation label
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'cifar100'
data_test_opt['split'] = 'test'

imgs_per_cat = data_train_opt['imgs_per_cat'] if ('imgs_per_cat' in data_train_opt) else None


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_feature_map(model,dloader):
    with torch.no_grad():

        features = []
        labels = []
        for batch in dloader(0):

            inputs, labels_batch = batch

            inputs = inputs.cuda() #Load input to cuda for fast inference

            model(inputs) #Feed data to bownet

            feature = model.resblock3_256b_fmaps #Get the feature map

        #Net to run flatten here
            feature = feature.cpu().numpy() #Need to transfer to numpy

            features.append(feature)



            # print(feature.shape)

            labels_batch = labels_batch.numpy()
            labels.append(labels_batch)
            # print(labels_batch.shape)
    features = np.concatenate(features)

    labels = np.concatenate(labels)

    return features,labels



dataset_train = GenericDataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'],
    random_sized_crop=data_train_opt['random_sized_crop'],
    num_imgs_per_cat=imgs_per_cat)
dataset_test = GenericDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'],
    random_sized_crop=data_test_opt['random_sized_crop'])

dloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=data_train_opt['batch_size'],
    unsupervised=data_train_opt['unsupervised'],
    epoch_size=data_train_opt['epoch_size'],
    num_workers=4,
    shuffle=True)

dloader_test = DataLoader(
    dataset=dataset_test,
    batch_size=data_test_opt['batch_size'],
    unsupervised=data_test_opt['unsupervised'],
    epoch_size=data_test_opt['epoch_size'],
    num_workers=4,
    shuffle=False)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = "bownet_checkpoint"
checkpoint = torch.load(PATH)

bownet,optimizer,epoch,loss = load_checkpoint(checkpoint,device)
criterion = nn.CrossEntropyLoss().to(device)


with torch.cuda.device(0):
    for para in bownet.parameters():
        para.requires_grad = False
    # for epoch in range(num_epochs):  # loop over the dataset multiple times

    print()
    print("TRAINING")
    running_loss = 0.0
    loss_100 = 0.0

    print("number of batch: ",len(dloader_train))

    accs = []

    feature_train, labels = get_feature_map(bownet,dloader_train)


    print(feature_train.shape)
    print(labels.shape)

    feature_train = feature_train.reshape((feature_train.shape[0],-1))

    print(feature_train.shape)


    # clf = LinearSVC(random_state=0, tol=1e-5)
    print("Run K-means")
    # cluster_ids_x, cluster_centers = kmeans(X=feature_train, num_clusters=200, distance='euclidean', device=torch.device('cuda:0'))
    #
    # clf.fit(feature_train,labels)
    sk_kmeans = KMeans(n_clusters=2048,max_iter=20, tol=0.0001).fit(feature_train)

    # kmeans = MiniBatchKMeans(n_clusters=2048,random_state=0,batch_size=128,max_iter=20,verbose=1).fit(feature_train)
    print(sk_kmeans.cluster_centers_.shape)

    # print(kmeans.cluster_centers_.shape)

    # print("EVALUATION")
    #
    # print("number of batch: ",len(dloader_test))
    #
    # running_loss = 0.0
    # accs = []
    # for idx, batch in enumerate(tqdm(dloader_test())): #We don't feed epoch to dloader_test because we want a random batch
    #     start_time = time.time()
    #     # get the inputs; data is a list of [inputs, labels]
    #     inputs, labels = batch
    #
    #     #Load data to GPU
    #     inputs, labels = inputs.cuda(), labels.cuda()
    #     time_load_data = time.time() - start_time
    #
    #
    #     # forward + backward + optimize
    #
    #
    #     logits, preds = bownet(inputs)
    #
    #     feature_test = bownet.resblock3_256b_fmaps
    #
    #     print(x.shape)
    #
    #     # print(preds[:,0])
    #
    #     #Compute loss
    #     loss = criterion(preds[:,0], labels)
    #
    #
    #     # print statistics
    #     running_loss += loss.item()
    #
    #     accs.append(accuracy(preds[:,0].data, labels, topk=(1,))[0].item())
    #
    #
    #
    #
    # # plt.imshow(check_input)
    # # plt.savefig("imag" + str(epoch) + ".png")
    # accs = np.array(accs)
    # print("epoche test accuracy: ",accs.mean())
    #
    # print("Time to load the data", time_load_data)
    # print("Time to finish an epoch ", time.time() - start_epoch)
    # print('[%d, %5d] epoches loss: %.3f' %
    #       (epoch, len(dloader_test), running_loss / len(dloader_test)))
