# RCAS
Private Pytorch implementation of [CLASS-SWAP LEARNING FOR CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS VIA REINITIALIZATION OF CLASS-AWARE SYNAPSES]


## Abstract
In continual learning research, the primary goal is to learn new knowledge while preventing catastrophic forgetting of previously trained knowledge. However, in dynamic industrial environments, certain learned classes are no longer required and need to adapt to new classes, as in semiconductor defect detection where certain defects are resolved while new ones emerge. To address this challenge, we present class-swap learning, an efficient fine-tuning method that rapidly learns the new classes by swapping the unused target classes. Our method especially focuses on conditional Generative Adversarial Networks (cGANs) that reinitializes important weights in generating the target classes by adapting a gradient-based algorithm. In our experimental results, class-swap learning shows faster convergence rates in learning the new classes when compared to the standard fine-tuning in terms of qualitative measurements.

## Methods
- **Overview of Class-swap learning for cGANs.**
<img src=https://github.com/mshdjren/RCAS/blob/master/figures/main_figure.jpg>

 1) Estimate the importannce $\Omega_{ij}^{g}$ , $\Omega_{ij}^{d}$ of each pre-trained generator and discriminator weights by calculating the gradients of each weight in inferring the images of target class.

 2) Select the important weight $w_{ij}^{g}$ , $w_{ij}^{d}$ above the arbitrary threshold for re-initialization.
  
 3) Re-initialize the selected weights $w_{ij}^{g}$ , $w_{ij}^{d}$ of pre-trained generator and discriminator to 0 for forgetting the target class.

## Results
- **Sample images when overwriting target class 9 with new class 0.**
<img src=https://github.com/mshdjren/RCAS/blob/master/figures/figure_sample.jpg>

  1) MNIST: digit 9 to digit 0

  2) FashionMNIST: Ankle boot to T-shirt/top

  3) CIFAR-10: truck to airplane

- **Different top-k ratio re-initialization for generator pre-trained on FashionMNIST.**
<img src=https://github.com/mshdjren/RCAS/blob/master/figures/figure_top_ratio.jpg>

## How to run our code
We borrowed most of the implementation of conditional generation framework from [PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) repository
and memory aware synapses from [MAS-Pytorch](https://github.com/deepanshgoyal33/MAS-Pytorch/tree/master) repository.

This code has been tested with Ubuntu 20.04, A100 GPUs with CUDA 12.2, Python 3.8, Pytorch 1.10.

- **Training (Class swap learning for target class)**
````
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -metrics is fid prdc -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -mas 
````
- **Training specific number of front layers of generator and discriminator (from FreezeD)**
````
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -metrics is fid prdc -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -mas -freezeD 
````
- **Different top-k ratio re-initialization"**

If re-initialize the top 50% weights based on importance, the `selectG_D_topk_ratio` variable is defined as 2 in the code as $\frac{100}{topk \textunderscore ratio}$.

````
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -t -metrics is fid prdc -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -mas -selectG_topk_ratio G_TOPK_RATIO -selectD_topk_ratio D_TOPK_RATIO
````
- **Testing (generating target/remaining classes images)**
````
CUDA_VISIBLE_DEVICES=0 python3 src/main.py -v -cfg CONFIG_PATH -ckpt CKPT -save SAVE_DIR
````

## License
This project is an open-source library under the MIT license (MIT). However, portions of the library are available under distinct license terms: StyleGAN2, StyleGAN2-ADA, and StyleGAN3 are licensed under [NVIDIA source code license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/LICENSE-NVIDIA), and PyTorch-FID is licensed under [Apache License](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/metrics/fid.py), same as mentioned by [PyTorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN). 
