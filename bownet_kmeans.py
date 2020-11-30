'''
2020 ML Reproducibility Challenge
Harry Nguyen, Stone Yun, Hisham Mohammad
Part of our submission for reproducing the CVPR 2020 paper: Learning Representations by Predicting Bags of Visual Words
https://arxiv.org/abs/2002.12247
'''
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import BowNet
from model import load_checkpoint
from dataloader import DataLoader, GenericDataset
from kmeans_pytorch import kmeans
# alternatively, we can use SKLearn
from sklearn.cluster import KMeans, MiniBatchKMeans

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
PATH = "bownet_checkpoint1.pt"
checkpoint = torch.load(PATH)

bownet,_,_,_ = load_checkpoint(checkpoint,device, BowNet)

dataset_train = GenericDataset(
    dataset_name='cifar100',
    split='train',
    random_sized_crop=False,
    num_imgs_per_cat=None)

dloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=128,
    unsupervised=False,
    epoch_size=None,
    num_workers=4,
    shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_RotNet_vocab(bownet: BowNet, K: int=2048):
    """After training RotNet, we will get the bag-of-words vocabulary using Euclidean distance based KMeans clustering"""
    feature_vectors_list = []

    # Iterate through trainset, compiling set of feature maps before performing KMeans clustering
    for i, data in enumerate(tqdm(dloader_train(0))):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data
            inputs = inputs.cuda()

            # Forward data through bownet, then get resblock_3_256b fmaps
            # The authors propose densely sampling feature C-dimensional feature vectors at each
            # spatial location where C is the size of the channel dimension. These feature vectors are used for the KMeans clustering
            outputs = bownet(inputs)
            #resblock3_fmaps = bownet.resblock3_256b_fmaps.transpose((0, 2, 3, 1))
            resblock3_fmaps = bownet.resblock3_256b_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))
            resblock3_feature_vectors = resblock3_fmaps.reshape(-1, resblock3_fmaps.shape[-1])
            feature_vectors_list.extend(resblock3_feature_vectors)

    rotnet_feature_vectors = np.array(feature_vectors_list)
    
    # Using MiniBatchKmeans because regular KMeans is too compute heavy
    start = time.time()
    sk_kmeans = MiniBatchKMeans(n_clusters=K, n_init=5, max_iter=100).fit(rotnet_feature_vectors)
    print(f"MiniBatchKMeans takes {time.time() - start}s")
    rotnet_vocab = sk_kmeans.cluster_centers_
    import pdb; pdb.set_trace()

    return sk_kmeans, rotnet_vocab

def train_bow_reconstruction(KMeans_vocab, K: int=2048):
    """Main training method presented in the paper. 
    Learning to reconstruct BOW histograms from perturbed images. Minimizes CrossEntropyLoss"""
# Just writing pseudo-code for now
    num_epochs = 200

    # Now for actual BoW training, output is equal to K
    bownet = BowNet(num_classes=K, bow_training=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(bownet.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dloader_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            # labels are now the expected BOW histogram rather than class label
            inputs, _ = data
            labels = KMeans_vocab.predict(inputs)
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, preds = bownet(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return

with torch.cuda.device(0):
    sk_kmeans, rotnet_vocab = build_RotNet_vocab(bownet)
    train_bow_reconstruction(sk_kmeans, K=2048)
