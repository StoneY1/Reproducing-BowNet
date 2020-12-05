# Reproducing-BowNet
Our reproducibility effort based on the 2020 ML Reproducibility Challenge. We are reproducing the results of this CVPR 2020 paper:
**Learning Representations by Predicting Bags of Visual Words** by Gidaris _et al_
S. Gidaris, A. Bursuc, N. Komodakis, P. Pérez, and M. Cord, “Learning Representations by Predicting Bags of Visual Words,” ArXiv, 27-Feb-2020. [Online]. Available: https://arxiv.org/abs/2002.12247. [Accessed: 15-Nov-2020]. 

Group project for UWaterloo course SYDE 671 - Advanced Image Processing by Harry Nguyen, Stone Yun, Hisham Mohammad

Code base is implemented with PyTorch. Dataloader is adapted from Github released by authors of the RotNet paper: https://github.com/gidariss/FeatureLearningRotNet

Our model definitions are in `model.py`. Custom loss and layer class definitions are in `layers.py`

See dependencies.txt for list of libraries that need to be installed. Pip install or conda install both work

Before running the experiments:

Inside the project code, create a folder `./datasets/CIFAR`, download the dataset CIFAR100 from https://www.cs.toronto.edu/~kriz/cifar.html and put in the folder.

## For running the code:
Pretrained weights of BowNet and RotNet from our best results are in `saved_weights` directory.
To generate your own RotNet checkpoint, running `rotation_prediction_training.py` will train a new RotNet from scratch. The checkpoint is saved as `rotnet1_checkpoint.pt`

To run `rotnet_linearclf.py` or `rotnet_nonlinearclf.py`, you need to have the checkpoint file of pretrained RotNet, download here (eg. saved_weights/rotnet.pt). These scripts load the pretrained RotNet and use its feature maps to train a classifier on CIFAR-100 prediction.

`$python rotnet_linearclf.py --checkpoint /path/to/checkpoint`

`$python rotnet_nonlinearclf.py --checkpoint /path/to/checkpoint`

`bownet_plus_linearclf_cifar_training.py` takes pretrained BowNet and uses feature maps to train linear classifier on CIFAR-100. `kmeans_cluster_and_bownet_training.py` loads pretrained RotNet, performs KMeans clustering of feature map, then trains BowNet on BOW reconstruction. Thus, you'll need pretrained BowNet and RotNet checkpoints respectively.

We also include a pre-computed RotNet codebook for K = 2048 clusters. If you include the path to it for `kmeans_cluster_and_bownet_training.py` the script will skip the codebook generation step and go straight to BOW reconstruction training

`$python bownet_plus_linearclf_cifar_training.py --checkpoint /path/to/bownet/checkpoint`

`$python kmeans_cluster_and_bownet_training.p --checkpoint /path/to/rotnet/checkpoint [optional: --rotnet_vocab /path/to/rotnet/vocab.npy]`
