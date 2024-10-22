# RCAS
Private Pytorch implementation of [CLASS-SWAP LEARNING FOR CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS VIA REINITIALIZATION OF CLASS-AWARE SYNAPSES]


## Abstract
In continual learning research, the primary goal is to learn new knowledge while preventing catastrophic forgetting of previously trained knowledge. However, in dynamic industrial environments, certain learned classes are no longer required and need to adapt to new classes, as in semiconductor defect detection where certain defects are resolved while new ones emerge. To address this challenge, we present classswap learning, an efficient fine-tuning method that rapidly learns the new classes by swaping the unused target classes.
Our method especially focuses on conditional Generative Adversarial Networks (cGANs) that re-initializes important weights in generating the target classes by adapting gradientbased algorithm. In our experimental results, class-swap learning shows faster convergence rate in learning the new classes when compared to standard fine-tuning in terms of qualitative measurements.

## Methods
- **Overview of Class-swap learning for cGANs.**
<img src=https://github.com/mshdjren/RCAS/blob/master/figures/main_figure.jpg> 

## Results
- **Sample images when overwriting target class 9 with new class 0.** 
<img src=https://github.com/mshdjren/RCAS/blob/master/figures/figure_sample.jpg>
MNIST: digit 9 to digit 0, FashionMNIST: Ankle boot to T-shirt/top and CIFAR-10: truck to airplane

- **Different top-k ratio re-initialization for generator pre-trained on FashionMNIST.**
<img src=https://github.com/mshdjren/RCAS/blob/master/figures/figure_top_ratio.jpg>

## Requirements:
$ git clone https://github.com/eriklindernoren/PyTorch-GAN

$ cd PyTorch-GAN/

$ sudo pip3 install -r requirements.txt

````
torch>=0.4.0
torchvision
matplotlib
numpy
scipy
pillow
urllib3
scikit-image
````
This code has been tested with Ubuntu 20.04, A100 GPUs with CUDA 12.2, Python 3.8, Pytorch 1.10.

## How to run our code
Our code built upon the repository of PyTorch-GAN.

We borrowed most of the implementation of conditional generation framework from PyTorch-GAN repository.

- **Training (Target-class unleraning for specific class)**
````
$ cd implementations/acgan/
$ python3 acgan.py
````

- **Testing (generating forgetting/remaining classes images)**
````
$ cd implementations/acgan/
$ python3 acgan.py
````

## License
A patent application for XMP has been submitted and is under review for registration. XMP is licensed under the CC-BY-NC-SA-4.0 license limiting any commercial use.

