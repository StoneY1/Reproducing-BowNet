from __future__ import print_function
import argparse
import os
import imp
from dataloader import DataLoader, GenericDataset, get_dataloader
import matplotlib.pyplot as plt


import copy
from model import BowNet, BowNet2, LinearClassifier, NonLinearClassifier
from utils import accuracy, load_checkpoint
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans



parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',  type=str, help='path to the checkpoint')
parser.add_argument('--fmap_depth', default=256,  type=int, help='Depth of fmaps used to train linear classifier')
args = parser.parse_args()

BOWNET_PATH = args.checkpoint
bownet_fmap_depth = args.fmap_depth

if args.checkpoint == None:
    sys.exit("Please include checkpoint with arg --checkpoint /path/to/checkpoint")

if args.fmap_depth != 256 and args.fmap_depth != 128:
    sys.exit("256 and 128 are the only valid fmap depth values")

# Set train and test datasets and the corresponding data loaders
batch_size = 128
K_clusters = 2048
bownet_fmap_size = 8 if bownet_fmap_depth == 256 else 16

dloader_train = get_dataloader('train', 'cifar', batch_size)
dloader_test = get_dataloader('test', 'cifar', batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LINEAR_CLF_PATH = f"bownet1_{bownet_fmap_depth}fmap_linearclf.pt"
bow_training = True
bownet, _, _, _ = load_checkpoint(BOWNET_PATH, device, BowNet, K_clusters, bow_training)

classifier = LinearClassifier(100, bownet_fmap_depth, bownet_fmap_size).to(device)
num_epochs = 400

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=10)

for para in bownet.parameters():
    para.requires_grad = False

with torch.cuda.device(0):

    classifier.train()
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
        bownet.eval()
        classifier.train()
        for idx, batch in enumerate(tqdm(dloader_train(epoch))): #We feed epoch in dloader_train to get a deterministic batch
        # for idx, batch in enumerate(dloader_train(epoch)):

            start_time = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            #Load data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()

            bow_logits, softmax_histograms = bownet(inputs)
            if bownet_fmap_depth == 256:
                bownet_fmaps = bownet.resblock3_256_fmaps
            elif bownet_fmap_depth == 128:
                bownet_fmaps = bownet.resblock2_128_fmaps
            else:
                raise ValueError(f"Wrong fmap depth for bownet features. Got {bownet_fmap_depth}")

            time_load_data = time.time() - start_time

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, preds = classifier(bownet_fmaps)

            #Compute loss
            loss = criterion(logits, labels)

            #Back Prop and Optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            loss_100 += loss.item()

            acc_batch, batch_correct_preds = accuracy(preds.data, labels, topk=(1,))
            total_correct += batch_correct_preds
            total_samples += preds.size(0)

        print("epoch training accuracy: ", 100*total_correct/total_samples)

        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
              (epoch, len(dloader_train), running_loss / len(dloader_train)))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, LINEAR_CLF_PATH)

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
        
        for idx, batch in enumerate(tqdm(dloader_test())): #We don't feed epoch to dloader_test because we want a random batch
            start_time = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            #Load data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()
            time_load_data = time.time() - start_time


            bow_logits, softmax_histograms = bownet(inputs)
            if bownet_fmap_depth == 256:
                bownet_fmaps = bownet.resblock3_256_fmaps
            elif bownet_fmap_depth == 128:
                bownet_fmaps = bownet.resblock2_128_fmaps
            else:
                raise ValueError(f"Wrong fmap depth for bownet features. Got {bownet_fmap_depth}")

            logits, preds = classifier(bownet_fmaps)

            #Compute loss
            loss = criterion(logits, labels)

            # print statistics
            running_loss += loss.item()

            acc_batch, batch_correct_preds = accuracy(preds.data, labels, topk=(1,))
            test_correct += batch_correct_preds
            test_total += preds.size(0)

        
        #lr_scheduler.step() # Use this if not using ReduceLROnPlateau scheduler
        lr_scheduler.step(running_loss/len(dloader_test)) # For LR scheduler that monitors test loss

        #print("epoche test accuracy: ",accs.mean())
        print("epoch test accuracy: ", 100*test_correct/test_total)

        print("Time to load the data", time_load_data)
        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
              (epoch, len(dloader_test), running_loss / len(dloader_test)))
