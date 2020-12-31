'''
2020 ML Reproducibility Challenge
Harry Nguyen, Stone Yun, Hisham Mohammad
Part of our submission for reproducing the CVPR 2020 paper: Learning Representations by Predicting Bags of Visual Words
https://arxiv.org/abs/2002.12247
'''
import numpy as np
import argparse
import time
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import BowNet, BowNet2, WRN_28_K
from utils import load_checkpoint
from layers import SoftCrossEntropyLoss
from dataloader import DataLoader, GenericDataset, get_dataloader
from sklearn.cluster import KMeans, MiniBatchKMeans

def build_RotNet_vocab(rotnet: BowNet, K: int=2048):
    """After training RotNet, we will get the bag-of-words vocabulary using Euclidean distance based KMeans clustering"""
    feature_vectors_list = []

    # Iterate through trainset, compiling set of feature maps before performing KMeans clustering
    rotnet.eval()
    for i, data in enumerate(tqdm(dloader_rotnet_vocab(0))):

        # get the inputs; data is a list of [inputs, labels]
        inputs, _ = data
        inputs = inputs.cuda()

        # Forward data through bownet, then get resblock_3_256b fmaps
        # The authors propose densely sampling feature C-dimensional feature vectors at each
        # spatial location where C is the size of the channel dimension. These feature vectors are used for the KMeans clustering
        outputs = rotnet(inputs)
        resblock3_fmaps = rotnet.resblock3_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))
        #resblock3_fmaps = rotnet.resblock3_256_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))
        resblock3_feature_vectors = resblock3_fmaps.reshape(-1, resblock3_fmaps.shape[-1])
        feature_vectors_list.extend(resblock3_feature_vectors)


    rotnet_feature_vectors = np.array(feature_vectors_list)

    # Using MiniBatchKmeans because regular KMeans is too compute heavy
    start = time.time()
    sk_kmeans = MiniBatchKMeans(n_clusters=K, n_init=10, max_iter=300, batch_size=512).fit(rotnet_feature_vectors)
    print(f"MiniBatchKMeans takes {time.time() - start}s")
    rotnet_vocab = sk_kmeans.cluster_centers_

    return sk_kmeans, rotnet_vocab

def initialize_kmeans_from_vocab(vocab_path, K: int=2048, num_features: int=256):
    """Loads cluster_centers from a NPY file and initializes an SKLearn.KMeans object with given BOW vocab.
    This way we won't need to regenerate a new codebook every time we want to run a BowNet experiment.
    """

    bow_vocab = np.load(vocab_path)
    test_data = np.random.uniform(0, 1, size=(K, num_features)).astype('float32')
    
    # A rather inelegant hack, but we need to call fit() in order to use the predict() method
    loaded_kmeans = MiniBatchKMeans(n_clusters=K, n_init=1, max_iter=1).fit(test_data)

    # Replace the cluster_centers_ with those loaded from the rotnet_vocab.npy
    loaded_kmeans.cluster_centers_ = bow_vocab

    return loaded_kmeans

def get_bow_histograms(KMeans_vocab, fmaps, spatial_density, K: int=2048):
    """Given a KMeans vocab and collection of fmaps, generates the associated BOW histogram"""
    fmap_vectors = fmaps.reshape(-1, fmaps.shape[-1])
    batch_pred_clusters = KMeans_vocab.predict(fmap_vectors).reshape(-1, spatial_density)
    bow_hist_labels = [np.histogram(pred_clusters, bins=K, range=(0, K), density=True)[0] for pred_clusters in batch_pred_clusters]
    return np.array(bow_hist_labels)

def train_bow_reconstruction(KMeans_vocab, dloader_train, dloader_test, rotnet, bownet, bownet_checkpoint_path, K: int=2048, fmap_size: int=8):
    """Main training method presented in the paper.
    Learning to reconstruct BOW representations from perturbed images. Minimizes SoftCrossEntropyLoss.
    We had to define our own CrossEntropy since the PyTorch version assums discrete classification.""" 

    num_epochs = 150

    # Freeze RotNet checkpoint
    for para in rotnet.parameters():
        para.required_grad = False

    rotnet.eval()
    criterion = torch.nn.MSELoss().to(device)
    #criterion = SoftCrossEntropyLoss().to(device)
    optimizer = optim.SGD(bownet.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.3)
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
            rotnet_logits, rotnet_preds = rotnet(label_imgs)
            #label_fmaps = rotnet.resblock3_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))
            #label_fmaps = rotnet.resblock3_256_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))

            bownet_logits, preds = bownet(inputs)
            # Not the cleanest way but not much choice given we don't have a CUDA based KMeans function
            #labels = get_bow_histograms(KMeans_vocab, label_fmaps, fmap_size**2, K)
            #labels = torch.Tensor(labels).cuda()
            #loss = criterion(logits, labels)
            loss = criterion(bownet_logits, rotnet_logits)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f"[***] Epoch {epoch} training loss: {running_loss/len(dloader_train)}")
        #import pdb; pdb.set_trace()

        torch.save({
            #'epoch': epoch,
            'model_state_dict': bownet.state_dict(),},
            #'optimizer_state_dict': optimizer.state_dict(),
            #'loss': loss,},
            bownet_checkpoint_path)

        torch.cuda.empty_cache()
        optimizer.zero_grad()
        bownet.eval()
        running_loss = 0.0
        for i, data in enumerate(tqdm(dloader_test())):
            inputs, label_imgs = data
            inputs = inputs.cuda()
            label_imgs = label_imgs.cuda()

            rotnet_logits, rotnet_preds = rotnet(label_imgs)
            #label_fmaps = rotnet.resblock3_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))
            #label_fmaps = rotnet.resblock3_256_fmaps.detach().cpu().numpy().transpose((0, 2, 3, 1))

            logits, preds = bownet(inputs)
            #labels = get_bow_histograms(KMeans_vocab, label_fmaps, fmap_size**2, K)
            #labels = torch.Tensor(labels).cuda()
            #loss = criterion(logits, labels)
            loss = criterion(logits, rotnet_logits)

            # print statistics
            running_loss += loss.item()
        print(f"[***] Epoch {epoch} test loss: {running_loss/len(dloader_test)}")

    print('Finished Training')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, help='path to the checkpoint')
    parser.add_argument('--rotnet_vocab',  type=str, help='path to precomputed RotNet vocab')
    args = parser.parse_args()
    
    ROTNET_PATH = args.checkpoint
    VOCAB_PATH = args.rotnet_vocab
    if args.checkpoint == None:
        sys.exit("Please include checkpoint with arg --checkpoint /path/to/checkpoint")

    # Need separate dataloader with mode == 'kmeans' for generating the BOW vocab of RotNet
    # This loader just feeds us the training images without the data augmentation
    dloader_rotnet_vocab = get_dataloader(split="train", mode="kmeans", batch_size=128)
    
    batch_size = 128
    #batch_size = 64
    K = 2048
    fmap_size = 8
    dloader_train = get_dataloader(split="train", mode="bow", batch_size=batch_size)
    dloader_test = get_dataloader(split="test", mode="bow", batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bownet_checkpoint_path = f'bownet1_K{K}_checkpoint.pt'
    rotnet, _, _, _ = load_checkpoint(ROTNET_PATH, device, BowNet)
    # Now for actual BoW training, output is equal to K
    bownet = copy.deepcopy(rotnet)
    #bownet = BowNet(num_classes=4, bow_training=False).to(device)
    #bownet = BowNet(num_classes=K, bow_training=True).to(device)

    with torch.cuda.device(0):
        vocab_path = f"{K}_RotNet1_BOW_Vocab.npy" if VOCAB_PATH is None else VOCAB_PATH
        if VOCAB_PATH is None:
            #sk_kmeans, rotnet_vocab = build_RotNet_vocab(rotnet, K)
            sk_kmeans = 0
            #np.save(vocab_path, rotnet_vocab)
        else:
            # If vocab is already generated, we can initialize a KMeans object and go straight to BOW training
            sk_kmeans = initialize_kmeans_from_vocab(vocab_path, K, num_features=256)
        train_bow_reconstruction(sk_kmeans, dloader_train, dloader_test, rotnet, bownet, bownet_checkpoint_path, K, fmap_size)
