


from __future__ import print_function
import argparse
import os
import imp
from dataloader import DataLoader, GenericDataset
import matplotlib.pyplot as plt

from model import BowNet
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

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

num_epochs = 200
bownet = BowNet(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(bownet.parameters(), lr=0.1, momentum=0.9)

# tensors = {}
# tensors['dataX'] = torch.FloatTensor()
# tensors['labels'] = torch.LongTensor()
with torch.cuda.device(0):
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        loss_100 = 0.0

        print("number of batch: ",len(dloader_train))
        start_epoch = time.time()
        for idx, batch in enumerate(tqdm(dloader_train(epoch))):
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
            loss = criterion(preds[:,0], labels)


            #Back Prop and Optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            loss_100 += loss.item()

            if idx % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch , idx, loss_100/100))
                loss_100 = 0.0
                acc = accuracy(preds[:,0].data, labels, topk=(1,))[0].item()
                print("accuracy 100 batch: ",acc)
                print("Time to finish 100 batch", time.time() - start_time)

        # plt.imshow(check_input)
        # plt.savefig("imag" + str(epoch) + ".png")
        acc = accuracy(preds[:,0].data, labels, topk=(1,))[0].item()
        print("epoches accuracy: ",acc)

        print("Time to load the data", time_load_data)
        print("Time to finish an epoch ", time.time() - start_epoch)
        print('[%d, %5d] epoches loss: %.3f' %
              (epoch, len(dloader_train), running_loss / len(dloader_train)))

        file_name = 'saved_model_state' + str(epoch) + '.pt'
        torch.save(bownet.state_dict(), file_name)


print('Finished Training')
