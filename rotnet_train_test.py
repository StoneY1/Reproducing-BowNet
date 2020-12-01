from __future__ import print_function
import argparse
import copy
import os
import imp
from dataloader import DataLoader, GenericDataset
import matplotlib.pyplot as plt

from model import BowNet
#from model import BowNet2 as BowNet
#from model import BowNet3 as BowNet
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

# Set train and test datasets and the corresponding data loaders
batch_size = 32

data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['mode'] = 'rotation'
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'cifar100'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['mode'] = 'rotation'
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
        correct_preds = copy.deepcopy(correct_k)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, correct_preds.int().item()

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
    mode=data_train_opt['mode'],
    epoch_size=data_train_opt['epoch_size'],
    num_workers=4,
    shuffle=True)

dloader_test = DataLoader(
    dataset=dataset_test,
    batch_size=data_test_opt['batch_size'],
    mode=data_test_opt['mode'],
    epoch_size=data_test_opt['epoch_size'],
    num_workers=4,
    shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 100
bownet = BowNet(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss().to(device)
#optimizer = optim.SGD(bownet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # used for BowNet2 and 3
#optimizer = optim.Adam(bownet.parameters(), lr=0.01)
optimizer = optim.SGD(bownet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10)

# tensors = {}
# tensors['dataX'] = torch.FloatTensor()
# tensors['labels'] = torch.LongTensor()
with torch.cuda.device(0):
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        print()
        print(f"TRAINING Epoch {epoch+1}")
        running_loss = 0.0
        loss_100 = 0.0
        total_correct = 0
        total_samples = 0

        print("number of batch: ",len(dloader_train))
        start_epoch = time.time()
        accs = []
        bownet.train()
        for idx, batch in enumerate(tqdm(dloader_train(epoch))): #We feed epoch in dloader_train to get a deterministic batch

            start_time = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            check_input = inputs[0].permute(1, 2, 0)

            #Load data to GPU
            inputs, labels = inputs.cuda(), labels.cuda()
            time_load_data = time.time() - start_time

            # print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, preds = bownet(inputs)

            # print(preds[:,0])

            #Compute loss
            loss = criterion(logits[:, 0], labels)

            #Back Prop and Optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            loss_100 += loss.item()
            acc_batch, batch_correct_preds = accuracy(preds[:,0].data, labels, topk=(1,))
            accs.append(acc_batch[0].item())
            total_correct += batch_correct_preds
            total_samples += preds.size(0) 
            
            '''
            if idx % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch , idx, loss_100/100))
                loss_100 = 0.0
                acc, correct = accuracy(preds[:,0].data, labels, topk=(1,))[0].item()
                print("accuracy 100 batch: ",acc)
                print("Time to finish 100 batch", time.time() - start_time)
            '''

        # plt.imshow(check_input)
        # plt.savefig("imag" + str(epoch) + ".png")
        accs = np.array(accs)
        #print("epoch training accuracy: ",accs.mean())
        #lr_scheduler.step()
        print("epoch training accuracy: ", 100*total_correct/total_samples)

        print("Time to load the data", time_load_data)
        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
              (epoch, len(dloader_train), running_loss / len(dloader_train)))

        PATH = "bownet_checkpoint1.pt"
        #PATH = "bownet_checkpoint2.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': bownet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
        torch.cuda.empty_cache()
        print()
        print(f"EVALUATION Epoch {epoch+1}")
        bownet.eval()
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


            # forward + backward + optimize
            logits, preds = bownet(inputs)

            # print(preds[:,0])

            #Compute loss
            loss = criterion(logits[:,0], labels)


            # print statistics
            running_loss += loss.item()

            acc_batch, batch_correct_preds = accuracy(preds[:,0].data, labels, topk=(1,))
            accs.append(acc_batch[0].item())
            test_correct += batch_correct_preds
            test_total += preds.size(0) 
            #accs.append(accuracy(preds[:,0].data, labels, topk=(1,))[0].item())


        # plt.imshow(check_input)
        # plt.savefig("imag" + str(epoch) + ".png")

        # lr scheduler will monitor test loss
        lr_scheduler.step(running_loss/len(dloader_test))
        accs = np.array(accs)
        #print("epoche test accuracy: ",accs.mean())
        print("epoch test accuracy: ", 100*test_correct/test_total)

        print("Time to load the data", time_load_data)
        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
              (epoch, len(dloader_test), running_loss / len(dloader_test)))


print('Finished Training')
