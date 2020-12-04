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
from layers import SoftCrossEntropyLoss
from dataloader import DataLoader, GenericDataset
from kmeans_pytorch import kmeans
# alternatively, we can use SKLearn
from sklearn.cluster import KMeans, MiniBatchKMeans

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# PATH = "best_rotnet_checkpoint1_7985acc.pt"
PATH = "best_bownet_checkpoint1_7285acc.pt"
checkpoint = torch.load(PATH)

rotnet,_,_,_ = load_checkpoint(checkpoint,device, BowNet)

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
    batch_size=128,
    mode='bow',
    epoch_size=None,
    num_workers=4,
    shuffle=True)

dloader_test = DataLoader(
    dataset=dataset_test,
    batch_size=128,
    mode='bow',
    epoch_size=None,
    num_workers=4,
    shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_RotNet_vocab(rotnet: BowNet, K: int=2048):
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
            outputs = rotnet(inputs)
            resblock3_fmaps = rotnet.resblock3_256b_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))
            print(resblock3_fmaps.shape)
            resblock3_feature_vectors = resblock3_fmaps.reshape(-1, resblock3_fmaps.shape[-1])

            print(resblock3_feature_vectors.shape)
            feature_vectors_list.extend(resblock3_feature_vectors)



    rotnet_feature_vectors = np.array(feature_vectors_list)
    print(rotnet_feature_vectors.shape)


    # Using MiniBatchKmeans because regular KMeans is too compute heavy
    start = time.time()
    sk_kmeans = MiniBatchKMeans(n_clusters=K, n_init=5, max_iter=100).fit(rotnet_feature_vectors)
    print(f"MiniBatchKMeans takes {time.time() - start}s")
    rotnet_vocab = sk_kmeans.cluster_centers_

    return sk_kmeans, rotnet_vocab

def get_bow_histograms(KMeans_vocab, fmaps, spatial_density, K: int=2048):
    """Given a KMeans vocab and collection of fmaps, generates the associated BOW histogram"""
    fmap_vectors = fmaps.reshape(-1, fmaps.shape[-1])
    batch_pred_clusters = KMeans_vocab.predict(fmap_vectors).reshape(-1, spatial_density)
    bow_hist_labels = [np.histogram(pred_clusters, bins=K, range=(0, K), density=True)[0] for pred_clusters in batch_pred_clusters]
    return np.array(bow_hist_labels)

def train_bow_reconstruction(KMeans_vocab, dloader_train, dloader_test, rotnet_checkpoint, K: int=2048):
    """Main training method presented in the paper.
    Learning to reconstruct BOW histograms from perturbed images. Minimizes CrossEntropyLoss"""
    num_epochs = 150
    checkpoint = 'bownet_bow_training_checkpoint.pt'

    # Loading frozen RotNet checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rotnet_ckpt = torch.load(rotnet_checkpoint)

    rotnet, _, _, _ = load_checkpoint(rotnet_ckpt, device, BowNet)
    for para in rotnet.parameters():
        para.required_grad = False

    # Now for actual BoW training, output is equal to K
    bownet = BowNet(num_classes=K, bow_training=True).to(device)
    rotnet.eval()
    criterion = SoftCrossEntropyLoss().to(device)
    optimizer = optim.SGD(bownet.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        bownet.train()
        running_loss = 0.0
        print_cnt = 0
        for i, data in enumerate(tqdm(dloader_train(epoch))):
            # get the inputs; data is a list of [inputs, labels]
            # labels are now the expected BOW histogram rather than class label
            inputs, label_imgs = data
            inputs = inputs.cuda()
            label_imgs = label_imgs.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # Get feature maps of RotNet generated from reference images
            rotnet(label_imgs)
            label_fmaps = rotnet.resblock3_256b_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))

            logits, preds = bownet(inputs)
            # Not the cleanest way but not much choice given we don't have a CUDA based KMeans function
            labels = get_bow_histograms(KMeans_vocab, label_fmaps, 64, K)
            labels = torch.Tensor(labels).cuda()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print_cnt += 1
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / (100*print_cnt)))
        print(f"[***] Epoch {epoch} training loss: {running_loss/len(dloader_train)}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': bownet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,},
            checkpoint)

        torch.cuda.empty_cache()
        optimizer.zero_grad()
        bownet.eval()
        running_loss = 0.0
        for i, data in enumerate(tqdm(dloader_test())):
            inputs, label_imgs = data
            inputs = inputs.cuda()
            label_imgs = label_imgs.cuda()

            rotnet(label_imgs)
            label_fmaps = rotnet.resblock3_256b_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))

            logits, preds = bownet(inputs)
            labels = get_bow_histograms(KMeans_vocab, label_fmaps, 64, K)
            labels = torch.Tensor(labels).cuda()
            loss = criterion(logits, labels)

            # print statistics
            running_loss += loss.item()
        print(f"[***] Epoch {epoch} test loss: {running_loss/len(dloader_test)}")

    print('Finished Training')
    return

with torch.cuda.device(0):
    sk_kmeans, rotnet_vocab = build_RotNet_vocab(rotnet)
    np.save("RotNet_BOW_Vocab.npy", rotnet_vocab)
    train_bow_reconstruction(sk_kmeans, dloader_train, dloader_test, PATH, K=2048)
