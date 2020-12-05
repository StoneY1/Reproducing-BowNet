# Reproducing-BowNet
Our submission to the 2020 ML Reproducibility Challenge. We are reproducing the results of this CVPR 2020 paper:
**Learning Representations by Predicting Bags of Visual Words** by Gidaris _et al_
S. Gidaris, A. Bursuc, N. Komodakis, P. Pérez, and M. Cord, “Learning Representations by Predicting Bags of Visual Words,” ArXiv, 27-Feb-2020. [Online]. Available: https://arxiv.org/abs/2002.12247. [Accessed: 15-Nov-2020]. 


Cifa dataset rotated in 0 90 180 and 270

https://drive.google.com/drive/folders/14JMAO0xeaFt7VCYoc0DhKEbia0F_GuVO?usp=sharing

Before running the experiments:

Inside the project code, create a folder `./datasets/CIFAR`, download the dataset CIFAR100 from https://www.cs.toronto.edu/~kriz/cifar.html and put in the folder.

Pretrained weights of BowNet and RotNet are in saved_weights directory,

To run `rotnet_linearclf.py` or `rotnet_nonlinearclf.py`, you need to have the checkpoint, download here (eg. rotnet.pt).

`$python rotnet_linearclf.py --checkpoint /path/to/checkpoint`

`$python rotnet_nonlinearclf.py --checkpoint /path/to/checkpoint`

To run `bownet_plus_linearclf_cifar_training.py` (Takes pretrained BowNet and trains linear classifier on CIFAR-100) or `kmeans_cluster_and_bownet_training.py` (Loads pretrained RotNet, performs KMeans clustering of feature map, then trains BowNet on BOW reconstruction), you also need to have the checkpoint, download here (eg. bownet.pt). We also include a pre-computed RotNet codebook for K = 2048 clusters. If you include the path to it for `kmeans_cluster_and_bownet_training.py` and the script will skip the codebook generation step and go straight to BOW reconstruction training

`$python bownet_plus_linearclf_cifar_training.py --checkpoint /path/to/checkpoint`

`$python kmeans_cluster_and_bownet_training.p --checkpoint /path/to/checkpoint [optional: --rotnet_vocab /path/to/rotnet/vocab.npy]`
