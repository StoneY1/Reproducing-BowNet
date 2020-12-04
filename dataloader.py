"""
Data-loader from original RotNet paper's code
Gidaris et. al https://arxiv.org/abs/1803.07728
https://github.com/gidariss/FeatureLearningRotNet

"""

from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import copy
import os
import errno
import numpy as np
import sys
import csv

from pdb import set_trace as breakpoint

# Set the paths of the datasets here.
_CIFAR_DATASET_DIR = './datasets/CIFAR'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class GenericDataset(data.Dataset):
    """Will contain CIFAR100 dataset here"""
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop

        # The num_imgs_per_cats input argument specifies the number
        # of training examples per category that would be used.
        # This input argument was introduced in order to be able
        # to use less annotated examples than what are available
        # in a semi-superivsed experiment. By default all the
        # available training examplers per category are being used.
        self.num_imgs_per_cat = num_imgs_per_cat

        # print(dataset_name)
        print("Prepare data for: ",split)


        ###CIFAR100

        # For CIFAR-100
        self.mean_pix = (0.5071, 0.4867, 0.4408)
        self.std_pix = (0.2675, 0.2565, 0.2761)

        if self.random_sized_crop:
            raise ValueError('The random size crop option is not supported for the CIFAR dataset')

        transform = []

        if (split != 'test'): #If load training data, the perform augmentation
            # transform.append(transforms.RandomCrop(32, padding=4))
            #transform.append(transforms.RandomHorizontalFlip())
            pass
        transform.append(lambda x: np.asarray(x))

        self.transform = transforms.Compose(transform)
        self.data = datasets.__dict__[self.dataset_name.upper()](
            _CIFAR_DATASET_DIR, train=self.split=='train',
            download=True, transform=self.transform)

        if num_imgs_per_cat is not None:
            self._keep_first_k_examples_per_category(num_imgs_per_cat)


    def _keep_first_k_examples_per_category(self, num_imgs_per_cat):
        print('num_imgs_per_category {0}'.format(num_imgs_per_cat))

        if self.dataset_name=='cifar100':
            labels = self.data.test_labels if (self.split=='test') else self.data.train_labels
            data = self.data.test_data if (self.split=='test') else self.data.train_data
            label2ind = buildLabelIndex(labels)
            all_indices = []
            for cat in label2ind.keys():
                label2ind[cat] = label2ind[cat][:num_imgs_per_cat]
                all_indices += label2ind[cat]
            all_indices = sorted(all_indices)
            data = data[all_indices]
            labels = [labels[idx] for idx in all_indices]
            if self.split=='test':
                self.data.test_labels = labels
                self.data.test_data = data
            else:
                self.data.train_labels = labels
                self.data.train_data = data

            label2ind = buildLabelIndex(labels)
            for k, v in label2ind.items():
                assert(len(v)==num_imgs_per_cat)

        elif self.dataset_name=='imagenet':
            raise ValueError('Keeping k examples per category has not been implemented for the {0}'.format(dname))
        elif self.dataset_name=='place205':
            raise ValueError('Keeping k examples per category has not been implemented for the {0}'.format(dname))
        else:
            raise ValueError('Not recognized dataset {0}'.format(dname))


    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2))).copy()
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img)).copy()
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2)).copy()
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

class DataLoader(object):
    """If flag is set to unsupervised, will generate the rotations and rotation-labels during training"""
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 mode='rotation',  #rotation, cifar,bow
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.mode = mode
        self.num_workers = num_workers
        self.split = self.dataset.split

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix

        my_transformations = [
            transforms.ToPILImage(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            # transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ]

        # If testing we won't use any transforms
        self.passthrough_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
            ])

        # if self.unsupervised:
        if (self.mode == 'rotation'):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_pix, std=std_pix)
            ])
        else:
            print("Compse transofrom supervised mode")
            if self.split == "test":
                print("Testing set has no transforms")
                self.transform = self.passthrough_transform
            else:
                self.transform = transforms.Compose(my_transformations)
                self.inv_transform = transforms.Compose([
                Denormalize(mean_pix, std_pix),
                lambda x: x.numpy() * 255.0,
                lambda x: x.transpose(1,2,0).astype(np.uint8),
                ])

        # else:
        #     print("Not implemeted yet")
            #Something for mode bow

    def get_iterator(self, epoch=0):
        # print("get iterator")
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        # if self.unsupervised:
        if (self.mode == 'rotation'):
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                # print("load function")
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                img0 = np.array(img0)
                # print(img0)
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90)),
                    self.transform(rotate_img(img0, 180)),
                    self.transform(rotate_img(img0, 270))
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)
                batch_size, rotations, channels, height, width = batch[0].size()
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                return batch
        # else: # supervised mode
        elif(self.mode == "cifar"):
            print("get iterator supervised mode")
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img =np.array(img)
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        else:

            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, _ = self.dataset[idx]
                img =np.array(img)
                label = copy.deepcopy(img)
                standardized_img = self.passthrough_transform(label) # Transform name is terrible, but basically it applies ToTensor() and Normalize
                img = self.transform(img)
                return img, standardized_img
                #label = img
                #img = self.transform(img)
                #return img, img

            _collate_fun = default_collate
            # print("Not implemeted yet")

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size


def get_dataloader(batch_size=128,mode='cifar'):

    dataset_train = GenericDataset(
        dataset_name='cifar100',
        split='train',
        random_sized_crop=False,
        num_imgs_per_cat=None)
    dataset_test = GenericDataset(
        dataset_name='cifar100',
        split='test',
        random_sized_crop=False)

    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        mode = mode,
        unsupervised=False,
        epoch_size=None,
        num_workers=4,
        shuffle=True)

    dloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        unsupervised=False,
        mode = mode,
        epoch_size=None,
        num_workers=4,
        shuffle=False)

    return dloader_train,dloader_test

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = GenericDataset('cifar100','train')
    dataloader = DataLoader(dataset, batch_size=8, unsupervised=False)

    for b in dataloader(0):
        data, label = b
        print(label)
        break

    # inv_transform = dataloader.inv_transform
    # for i in range(data.size(0)):
    #     plt.subplot(data.size(0)/4,4,i+1)
    #     fig=plt.imshow(inv_transform(data[i]))
    #     fig.axes.get_xaxis().set_visible(False)
    #     fig.axes.get_yaxis().set_visible(False)

    plt.show()
