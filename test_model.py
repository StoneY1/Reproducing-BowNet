
from __future__ import print_function
import argparse
import copy
import os
import imp
from dataloader import DataLoader, GenericDataset, get_dataloader
import matplotlib.pyplot as plt

from model import BowNet2 as BowNet
from utils import load_checkpoint, accuracy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

# Set train and test datasets and the corresponding data loaders
batch_size = 64

dloader_test = get_dataloader('test', 'rotation', batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 200
criterion = nn.CrossEntropyLoss().to(device)

PATH = "rotnet2_checkpoint.pt"
#PATH = "best_rotnet_checkpoint1_7985acc.pt"

bownet,_,_,_ = load_checkpoint(PATH, device, BowNet)
bownet.eval()
with torch.cuda.device(0):
    print(f"EVALUATION")

    print("number of batch: ",len(dloader_test))
    start_epoch = time.time()
    running_loss = 0.0
    accs = []
    test_correct = 0
    test_total = 0
    epoch = 1
    for idx, batch in enumerate(tqdm(dloader_test())): #We don't feed epoch to dloader_test because we want a random batch
        start_time = time.time()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch

        #Load data to GPU
        inputs, labels = inputs.cuda(), labels.cuda()
        time_load_data = time.time() - start_time

        # forward + backward + optimize
        logits, preds = bownet(inputs)

        #Compute loss
        loss = criterion(logits, labels)

        # print statistics
        running_loss += loss.item()

        acc_batch, batch_correct_preds = accuracy(preds.data, labels, topk=(1,))
        accs.append(acc_batch[0].item())
        test_correct += batch_correct_preds
        test_total += preds.size(0) 

    accs = np.array(accs)
    #print("epoche test accuracy: ",accs.mean())
    print("epoch test accuracy: ", 100*test_correct/test_total)

    print("Time to load the data", time_load_data)
    print("Time to finish an epoch ", time.time() - start_epoch)
    print('[%d, %5d] epoches loss: %.3f' %
          (epoch, len(dloader_test), running_loss / len(dloader_test)))


print('Finished Training')
