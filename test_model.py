


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
from model import load_checkpoint
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

# Set train and test datasets and the corresponding data loaders
batch_size = 128

data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = True
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'cifar100'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = True
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

dataset_test = GenericDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'],
    random_sized_crop=data_test_opt['random_sized_crop'])

dloader_test = DataLoader(
    dataset=dataset_test,
    batch_size=data_test_opt['batch_size'],
    unsupervised=data_test_opt['unsupervised'],
    epoch_size=data_test_opt['epoch_size'],
    num_workers=4,
    shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
num_epochs = 200
bownet = BowNet(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss().to(device)


PATH = "best_bownet1_checkpoint.pt"
checkpoint = torch.load(PATH, map_location=torch.device('cpu'))

bownet,_,_,_ = load_checkpoint(checkpoint,device, BowNet)
#with device:
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
    #inputs, labels = inputs.cuda(), labels.cuda()
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
accs = np.array(accs)
#print("epoche test accuracy: ",accs.mean())
print("epoch test accuracy: ", 100*test_correct/test_total)

print("Time to load the data", time_load_data)
print("Time to finish an epoch ", time.time() - start_epoch)
print('[%d, %5d] epoches loss: %.3f' %
      (epoch, len(dloader_test), running_loss / len(dloader_test)))


print('Finished Training')
