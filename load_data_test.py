from __future__ import print_function
import argparse
import os
import imp
from dataloader import DataLoader, GenericDataset
import matplotlib.pyplot as plt

# Set train and test datasets and the corresponding data loaders
batch_size = 128

data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = False
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'cifar100'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = False
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'cifar100'
data_test_opt['split'] = 'test'

imgs_per_cat = data_train_opt['imgs_per_cat'] if ('imgs_per_cat' in data_train_opt) else None

print(imgs_per_cat)

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

for b in dloader_train(0):
  data,label = b
  print("data: ",data.shape)
  print(data[0].shape)
  plt.imshow(data[0].permute(1, 2, 0))
  plt.show()
  print("label: ",label.shape)
  print(label[0])
  break
