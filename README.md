# Reproducing-BowNet
Our submission to the 2020 ML Reproducibility Challenge. We are reproducing the results of this CVPR 2020 paper:
**Learning Representations by Predicting Bags of Visual Words** by Gidaris _et al_
S. Gidaris, A. Bursuc, N. Komodakis, P. Pérez, and M. Cord, “Learning Representations by Predicting Bags of Visual Words,” ArXiv, 27-Feb-2020. [Online]. Available: https://arxiv.org/abs/2002.12247. [Accessed: 15-Nov-2020]. 


Cifa dataset rotated in 0 90 180 and 270

https://drive.google.com/drive/folders/14JMAO0xeaFt7VCYoc0DhKEbia0F_GuVO?usp=sharing

Before running the experiments:

Inside the project code, create a folder `./datasets/CIFAR`, download the dataset CIFAR100 from https://www.cs.toronto.edu/~kriz/cifar.html and put in the folder.

To run `rotnet_linearclf.py` or `rotnet_nonlinearclf.py`, you need to have the checkpoint, download here.

`$python rotnet_linearclf.py --checkpoint /path/to/checkpoint`

`$python rotnet_nonlinearclf.py --checkpoint /path/to/checkpoint`
