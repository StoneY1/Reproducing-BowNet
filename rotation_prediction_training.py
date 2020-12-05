from __future__ import print_function
import argparse
import copy
import os
import imp
from dataloader import DataLoader, GenericDataset, get_dataloader
import matplotlib.pyplot as plt

from model import BowNet
from utils import load_checkpoint, accuracy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

# Get train and test dataloaders
batch_size = 64
dloader_train = get_dataloader(split='train', mode='rotation', batch_size=batch_size)
dloader_test = get_dataloader(split='test', mode='rotation', batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = "rotnet1_checkpoint.pt"

#rotnet, optimizer, start_epoch, loss = load_checkpoint(PATH, device, BowNet)
num_epochs = 150
start_epoch = 0
end_epoch = num_epochs - start_epoch
rotnet = BowNet(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(rotnet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10)

with torch.cuda.device(0):
    for epoch in np.arange(start_epoch, end_epoch):  # loop over the dataset multiple times

        print()
        print(f"TRAINING Epoch {epoch+1}")
        running_loss = 0.0
        loss_100 = 0.0
        total_correct = 0
        total_samples = 0

        print("number of batch: ",len(dloader_train))
        start_epoch = time.time()
        accs = []
        rotnet.train()
        for idx, batch in enumerate(tqdm(dloader_train(epoch))): #We feed epoch in dloader_train to get a deterministic batch

            start_time = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            check_input = inputs[0].permute(1, 2, 0)

            #Load data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()
            time_load_data = time.time() - start_time

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, preds = rotnet(inputs)

            #Compute loss
            loss = criterion(logits, labels)

            #Back Prop and Optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            loss_100 += loss.item()
            acc_batch, batch_correct_preds = accuracy(preds.data, labels, topk=(1,))
            accs.append(acc_batch[0].item())
            total_correct += batch_correct_preds
            total_samples += preds.size(0) 
            
        accs = np.array(accs)
        print("epoch training accuracy: ", 100*total_correct/total_samples)

        print("Time to load the data", time_load_data)
        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
              (epoch, len(dloader_train), running_loss / len(dloader_train)))

        torch.save({
            'epoch': epoch,
            'model_state_dict': rotnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
        torch.cuda.empty_cache()
        print()
        print(f"EVALUATION Epoch {epoch+1}")
        rotnet.eval()
        print("number of batch: ",len(dloader_test))
        start_epoch = time.time()
        running_loss = 0.0
        accs = []
        test_correct = 0
        test_total = 0
        for idx, batch in enumerate(tqdm(dloader_test())): #We don't feed epoch to dloader_test because we want a random batch
            start_time = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            #Load data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()
            time_load_data = time.time() - start_time
            logits, preds = rotnet(inputs)

            #Compute loss
            loss = criterion(logits, labels)

            # print statistics
            running_loss += loss.item()

            acc_batch, batch_correct_preds = accuracy(preds.data, labels, topk=(1,))
            accs.append(acc_batch[0].item())
            test_correct += batch_correct_preds
            test_total += preds.size(0) 

        # lr scheduler will monitor test loss
        lr_scheduler.step(running_loss/len(dloader_test))
        accs = np.array(accs)
        print("epoch test accuracy: ", 100*test_correct/test_total)

        print("Time to load the data", time_load_data)
        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
              (epoch, len(dloader_test), running_loss / len(dloader_test)))


print('Finished Training')
