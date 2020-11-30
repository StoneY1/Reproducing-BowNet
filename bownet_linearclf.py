from __future__ import print_function
import argparse
import os
import imp
from dataloader import DataLoader, GenericDataset
import matplotlib.pyplot as plt

from model import BowNet3 as BowNet
from model import load_checkpoint, LinearClassifier, NonLinearClassifier
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
#from kmeans_pytorch import kmeans

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
print(device)
PATH = "bownet_checkpoint3.pt"
checkpoint = torch.load(PATH)

bownet,_,_,_ = load_checkpoint(checkpoint,device, BowNet)


classifier = NonLinearClassifier(100, 64, 16).to(device)
#classifier = LinearClassifier(100, 256, 8).to(device)
num_epochs = 200

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)
#optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

for para in bownet.parameters():
    para.requires_grad = False

with torch.cuda.device(0):

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        print()
        print("TRAINING")
        running_loss = 0.0
        loss_100 = 0.0

        print("number of batch: ",len(dloader_train))
        start_epoch = time.time()
        accs = []

        true_count = 0.0
        total_count = 0
        
        # Need to set bownet to evaluate so that it uses frozen BatchNorm params and no Dropout
        bownet.evaluate()
        # for idx, batch in enumerate(tqdm(dloader_train(epoch))): #We feed epoch in dloader_train to get a deterministic batch
        for idx, batch in enumerate(dloader_train(epoch)):

            start_time = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            # check_input = inputs[0].permute(1, 2, 0)


            #Load data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()

            bownet(inputs)
            conv_out = bownet.resblock1_64c_fmaps
            #conv_out = bownet.resblock2_128d_fmaps
            #conv_out = bownet.resblock3_256f_fmaps

            # print(conv_out.shape)


            time_load_data = time.time() - start_time

            # print(labels)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, preds = classifier(conv_out)

            # print(preds[:,0])

            #Compute loss
            loss = criterion(logits[:, 0], labels)
            predicts = torch.argmax(preds[:,0], dim=1)
            # predicts = predicts.reshape(1,-1)
            # print(labels.shape)
            # print(predicts.shape)
            # print(preds[:,0].shape)


            #Back Prop and Optimize
            loss.backward()
            optimizer.step()

            # print(classifier.fc1_out.weight.grad)

            # print statistics
            running_loss += loss.item()

            loss_100 += loss.item()

            # print(torch.argmax(preds[:,0], dim=1))
            # print(labels)
            #
            # print((predicts == labels).float().sum())
            # print(predicts)
            # print(labels)

            # accuracy(preds[:,0].data, labels, topk=(1,))[0].item()

            # accs.append(accuracy(preds[:,0].data, labels, topk=(1,))[0].item())
            # acc = accuracy(preds[:,0].data, labels, topk=(1,))[0].item()
            # print("accuracy 100 batch: ",acc)
            true_count += ((predicts == labels).float().sum()).cpu().numpy()
            total_count += predicts.size(0)


            if idx % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch , idx, loss_100/100))
                loss_100 = 0.0
                # acc = (((predicts == labels).float().sum())/128).cpu().numpy()
                # print("accuracy after 100 batch: ",acc)
                print("Time to finish 100 batch", time.time() - start_time)
        lr_scheduler.step()
        print("epoch acc ", 100*true_count/total_count)
        # plt.imshow(check_input)
        # plt.savefig("imag" + str(epoch) + ".png")
        # accs = np.array(accs)
        # print("epoch training accuracy: ",accs.mean())
        #
        # print("Time to load the data", time_load_data)
        # print("Time to finish an epoch ", time.time() - start_epoch)
        # print('[%d, %5d] epoches loss: %.3f' %
        #       (epoch, len(dloader_train), running_loss / len(dloader_train)))

        print("EVALUATION")
        #
        print("number of batch: ",len(dloader_test))
        #
        running_loss = 0.0
        test_correct = 0 
        test_total = 0
        accs = []
        for idx, batch in enumerate(tqdm(dloader_test())): #We don't feed epoch to dloader_test because we want a random batch
            start_time = time.time()
        #     # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
        #
        #     #Load data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()
            time_load_data = time.time() - start_time
        #
        #
        #     # forward + backward + optimize
        #
        #
            bownet_outputs = bownet(inputs)
        #
            feature_test = bownet.resblock1_64c_fmaps
            #feature_test = bownet.resblock2_128d_fmaps
            #feature_test = bownet.resblock3_256f_fmaps
            logits, preds = classifier(feature_test)
            predicts = torch.argmax(preds[:,0], dim=1)
             
            test_correct += ((predicts == labels).float().sum()).cpu().numpy()
            test_total += predicts.size(0)
        #
        #     print(x.shape)
        #
        #     # print(preds[:,0])
        #
        #     #Compute loss
            loss = criterion(logits[:,0], labels)
        #
        #
        #     # print statistics
            running_loss += loss.item()
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
        print("epoche test accuracy: ", 100 * test_correct/test_total)
        #
        # print("Time to load the data", time_load_data)
        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
               (epoch, len(dloader_test), running_loss / len(dloader_test)))
