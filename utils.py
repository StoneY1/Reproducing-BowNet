import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def load_checkpoint(checkpoint_path, device, bownet_arch, num_classes=4, bow_training=False):
    checkpoint = torch.load(checkpoint_path)
    bownet = bownet_arch(num_classes, bow_training).to(device)
    optimizer = optim.SGD(bownet.parameters(), lr=0.1, momentum=0.9,weight_decay= 5e-4)

    bownet.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']
    epoch = 0
    loss = 0

    return bownet, optimizer, epoch, loss

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
