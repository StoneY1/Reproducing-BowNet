from __future__ import print_function
import argparse
import os
import imp
from dataloader import get_dataloader,DataLoader, GenericDataset
import matplotlib.pyplot as plt


import copy
from model import BowNet,LinearClassifier, NonLinearClassifier
from utils import load_checkpoint, accuracy

#from model import BowNet3 as BowNet
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from itertools import chain

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
#from kmeans_pytorch import kmeans

# Set train and test datasets and the corresponding data loaders

batch_size = 128
dloader_train = get_dataloader('train', 'cifar', batch_size)
dloader_test = get_dataloader('test', 'cifar', batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# bownet,_,_,_ = load_checkpoint(checkpoint,device,BowNet)
rotnet = BowNet(100).to(device)
classifier = LinearClassifier(100).to(device)
#classifier = LinearClassifier(100, 256, 8).to(device)
num_epochs = 400

criterion = nn.CrossEntropyLoss().to(device)

all_params = chain(rotnet.parameters(), classifier.parameters())
optimizer = optim.SGD(list(rotnet.parameters()) + list(classifier.parameters()), lr=0.001, momentum=0.9, weight_decay=1e-6)


#optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

# for para in bownet.parameters():
#     para.requires_grad = False
#
# bownet.eval()
with torch.cuda.device(0):


    for epoch in range(num_epochs):  # loop over the dataset multiple times

        print()
        print("TRAINING")
        running_loss = 0.0
        loss_100 = 0.0

        print("number of batch: ",len(dloader_train))
        start_epoch = time.time()
        accs = []
        total_correct = 0
        total_samples = 0
        # Need to set bownet to evaluate so that it uses frozen BatchNorm params and no Dropout
        classifier.train()
        rotnet.train()
        for idx, batch in enumerate(tqdm(dloader_train(epoch))): #We feed epoch in dloader_train to get a deterministic batch
        # for idx, batch in enumerate(dloader_train(epoch)):

            start_time = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            # check_input = inputs[0].permute(1, 2, 0)


            #Load data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()

            rotnet(inputs)
            conv_out = rotnet.resblock3_256_fmaps

            # print(conv_out.shape)


            time_load_data = time.time() - start_time

            # print(labels.shape)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, preds = classifier(conv_out)

            # print(preds[:,0])

            #Compute loss
            loss = criterion(logits, labels)

            #Back Prop and Optimize
            loss.backward()
            optimizer.step()

            # print(classifier.fc1_out.weight.grad)

            # print statistics
            running_loss += loss.item()

            loss_100 += loss.item()

            acc_batch, batch_correct_preds = accuracy(preds.data, labels, topk=(1,))
            accs.append(acc_batch[0].item())
            total_correct += batch_correct_preds
            total_samples += preds.size(0)

            # if idx % 100 == 99:
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch , idx, loss_100/100))
            #     loss_100 = 0.0
            #     acc = accuracy(logits[:,0], labels, topk=(1,))[0].item()
            #     print("accuracy after 100 batch: ",acc)
            #     print("Time to finish 100 batch", time.time() - start_time)

        accs = np.array(accs)
        print("epoch training accuracy: ", 100*total_correct/total_samples)

        print("Time to load the data", time_load_data)
        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
              (epoch, len(dloader_train), running_loss / len(dloader_train)))

        print()
        torch.cuda.empty_cache()
        print("EVALUATION")

        print("number of batch: ",len(dloader_test))

        start_epoch = time.time()
        running_loss = 0.0
        accs = []
        test_correct = 0
        test_total = 0

        classifier.eval()
        rotnet.eval()
        for idx, batch in enumerate(tqdm(dloader_test())): #We don't feed epoch to dloader_test because we want a random batch
            start_time = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            #Load data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()
            time_load_data = time.time() - start_time


            # forward + backward + optimize
            rotnet(inputs)
            conv_out = rotnet.resblock3_256_fmaps

            logits, preds = classifier(conv_out)


            #Compute loss
            loss = criterion(logits, labels)


            # print statistics
            running_loss += loss.item()

            acc_batch, batch_correct_preds = accuracy(preds.data, labels, topk=(1,))
            accs.append(acc_batch[0].item())
            test_correct += batch_correct_preds
            test_total += preds.size(0)


        lr_scheduler.step() # Use this if not using ReduceLROnPlateau scheduler
        # lr_scheduler.step(running_loss/len(dloader_test))
        accs = np.array(accs)
        #print("epoche test accuracy: ",accs.mean())
        print("epoch test accuracy: ", 100*test_correct/test_total)

        print("Time to load the data", time_load_data)
        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
              (epoch, len(dloader_test), running_loss / len(dloader_test)))

        # file_name = "rotnet_linearclf_" + str(epoch) +"_" + str(100*test_correct/test_total) + ".pt"
        # PATH = "./rotnet_linear_ckpt/" + file_name
        # #PATH = "bownet_checkpoint2.pt"
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': classifier.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss,
        #     }, PATH)
