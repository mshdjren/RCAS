# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/evaluate.py

from argparse import ArgumentParser
import os
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.backends import cudnn
from PIL import Image
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import numpy as np
import pickle

import utils.misc as misc
import metrics.preparation as pp
import metrics.features as features
import metrics.ins as ins
import metrics.fid as fid
import metrics.prdc as prdc

# importing package
import matplotlib.pyplot as plt
import numpy as np

def prepare_evaluation():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("-metrics", "--eval_metrics", nargs='+', default=['fid'],
                        help="evaluation metrics to use during training, a subset list of ['fid', 'is', 'prdc'] or none")
    # parser.add_argument("--post_resizer", type=str, default="legacy", help="which resizer will you use to evaluate GANs\
    #                     in ['legacy', 'clean', 'friendly']")
    # parser.add_argument('--eval_backbone', type=str, default='InceptionV3_tf',\
    #                     help="[InceptionV3_tf, InceptionV3_torch, ResNet50_torch, SwAV_torch, DINO_torch, Swin-T_torch]")
    # parser.add_argument("--dset1", type=str, default=None, help="specify the directory of the folder that contains dset1 images (real).")
    # parser.add_argument("--dset1_feats", type=str, default=None, help="specify the path of *.npy that contains features of dset1 (real). \
    #                     If not specified, StudioGAN will automatically extract feat1 using the whole dset1.")
    # parser.add_argument("--dset1_moments", type=str, default=None, help="specify the path of *.npy that contains moments (mu, sigma) of dset1 (real). \
    #                     If not specified, StudioGAN will automatically extract moments using the whole dset1.")
    # parser.add_argument("--dset2", type=str, default=None, help="specify the directory of the folder that contains dset2 images (fake).")
    # parser.add_argument("--batch_size", default=256, type=int, help="batch_size for evaluation")

    parser.add_argument("--seed", type=int, default=-1, help="seed for generating random numbers")
    # parser.add_argument("-DDP", "--distributed_data_parallel", action="store_true")
    # parser.add_argument("--backend", type=str, default="nccl", help="cuda backend for DDP training \in ['nccl', 'gloo']")
    parser.add_argument("-tn", "--total_nodes", default=1, type=int, help="total number of nodes for training")
    # parser.add_argument("-cn", "--current_node", default=0, type=int, help="rank of the current node")
    # parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # if args.dset1_feats == None and args.dset1_moments == None:
    #     assert args.dset1 != None, "dset1 should be specified!"
    # if "fid" in args.eval_metrics:
    #     assert args.dset1 != None or args.dset1_moments != None, "Either dset1 or dset1_moments should be given to compute FID."
    # if "prdc" in args.eval_metrics:
    #     assert args.dset1 != None or args.dset1_feats != None, "Either dset1 or dset1_feats should be given to compute PRDC."

    gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
    world_size = gpus_per_node * args.total_nodes
    if args.seed == -1: args.seed = random.randint(1, 4096)
    if world_size == 1: print("You have chosen a specific GPU. This will completely disable data parallelism.")
    return args, world_size, gpus_per_node, rank


#Constrained to CIFAR10..?
args, world_size, gpus_per_node, rank = prepare_evaluation()

#IS
IS_dict = {}
#FID
FID_dict = {}
#intra-FID
iFID_remain_dict = {}
iFID_target_dict = {}

# 0step...(target-ifid)
# MNIST = 78.845
# FashionMNIST = 172.279
# CIFAR10 = 121.01

pre_train_target_iFID = 78.845
split = 1+10
#FOR fASHIONNNIST
naive_folder = "mas_fine_target_1_find"
#ELSE
# naive_folder = "mas_fine_target_1_last"


fine_folder = "mas_fine_target_1_last"


data = "MNIST"
topg_2_path = "MNIST-deep_conv_aux-MNIST_g_ablation_topg_2_topd_200_lr_0.0001_d_up_2-2023_09_03_23_50_22"
topg_5_path = "MNIST-deep_conv_aux-MNIST_gonly_topg_5_topd_200_lr_0.0001_d_up_2-2023_09_03_10_59_57"
topg_10_path = "MNIST-deep_conv_aux-MNIST_g_ablation_topg_10_topd_200_lr_0.0001_d_up_2-2023_09_03_23_50_29"
topg_100_path = "MNIST-deep_conv_aux-MNIST_g_ablation_topg_100_topd_200_lr_0.0001_d_up_2-2023_09_03_23_50_36"
topg_200_path = "MNIST-deep_conv_aux-MNIST_gonly_topg_5_topd_200_lr_0.0001_d_up_2-2023_09_03_10_59_57"
topg_1000_path = "MNIST-deep_conv_aux-MNIST_gonly_topg_5_topd_1000_lr_0.0001_d_up_2-2023_09_03_10_59_15"


if data == "MNIST":
    save_freq = 1000
elif data == "FashionMNIST":
    save_freq = 2000
else:
    save_freq = 4000


for i in range(1):
    metrics = np.load("./results/final/{folder}/{data}/statistics/{topg_2_path}/train/metrics.npy".format(folder=fine_folder, data=data, topg_2_path=topg_2_path), allow_pickle=True)
    iFID_remain = np.load("./results/final/{folder}/{data}/statistics/{topg_2_path}/iFID_remain.npy".format(folder=fine_folder, data=data, topg_2_path=topg_2_path), allow_pickle=True)
    iFID_target = np.load("./results/final/{folder}/{data}/statistics/{topg_2_path}/iFID_target.npy".format(folder=fine_folder, data=data, topg_2_path=topg_2_path), allow_pickle=True)

    methods = "topg_2"

    IS_dict[methods] = metrics.item().get("IS")
    FID_dict[methods] = metrics.item().get("FID")
    iFID_remain_dict[methods] = iFID_remain.item().get("remain_ifid")
    iFID_target_dict[methods] = iFID_target.item().get("target_ifid")

    print(IS_dict)
    print(FID_dict)
    print(iFID_remain_dict)
    print(iFID_target_dict)

 
    # max_IS
    # min_FID
    # min_iFID_remain
    # MIN_IFID

    
    # print("{i}th model IS convergence index: {iter} iterations".format(i="naive", iter = save_freq * IS_list.index(max_IS)))
    # print("{i}th model FID convergence index: {iter} iterations".format(i="naive", iter = save_freq * FID_list.index(min_FID)))
    # print("{i}th model intra_fid_remain convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_remain_list.index(min_iFID_remain)))
    # print("{i}th model intra_fid_target convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_target_list.index(min_iFID_target)))

for i in range(1):
    metrics = np.load("./results/final/{folder}/{data}/statistics/{topg_5_path}/train/metrics.npy".format(folder=fine_folder, data=data, topg_5_path=topg_5_path), allow_pickle=True)
    iFID_remain = np.load("./results/final/{folder}/{data}/statistics/{topg_5_path}/iFID_remain.npy".format(folder=fine_folder, data=data, topg_5_path=topg_5_path), allow_pickle=True)
    iFID_target = np.load("./results/final/{folder}/{data}/statistics/{topg_5_path}/iFID_target.npy".format(folder=fine_folder, data=data, topg_5_path=topg_5_path), allow_pickle=True)

    methods = "topg_5"

    IS_dict[methods] = metrics.item().get("IS")
    FID_dict[methods] = metrics.item().get("FID")
    iFID_remain_dict[methods] = iFID_remain.item().get("remain_ifid")
    iFID_target_dict[methods] = iFID_target.item().get("target_ifid")

    print(IS_dict)
    print(FID_dict)
    print(iFID_remain_dict)
    print(iFID_target_dict)

 
    # max_IS
    # min_FID
    # min_iFID_remain
    # MIN_IFID

    
    # print("{i}th model IS convergence index: {iter} iterations".format(i="naive", iter = save_freq * IS_list.index(max_IS)))
    # print("{i}th model FID convergence index: {iter} iterations".format(i="naive", iter = save_freq * FID_list.index(min_FID)))
    # print("{i}th model intra_fid_remain convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_remain_list.index(min_iFID_remain)))
    # print("{i}th model intra_fid_target convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_target_list.index(min_iFID_target)))


for i in range(1):
    metrics = np.load("./results/final/{folder}/{data}/statistics/{topg_10_path}/train/metrics.npy".format(folder=fine_folder, data=data, topg_10_path=topg_10_path), allow_pickle=True)
    iFID_remain = np.load("./results/final/{folder}/{data}/statistics/{topg_10_path}/iFID_remain.npy".format(folder=fine_folder, data=data, topg_10_path=topg_10_path), allow_pickle=True)
    iFID_target = np.load("./results/final/{folder}/{data}/statistics/{topg_10_path}/iFID_target.npy".format(folder=fine_folder, data=data, topg_10_path=topg_10_path), allow_pickle=True)

    methods = "topg_10"

    IS_dict[methods] = metrics.item().get("IS")
    FID_dict[methods] = metrics.item().get("FID")
    iFID_remain_dict[methods] = iFID_remain.item().get("remain_ifid")
    iFID_target_dict[methods] = iFID_target.item().get("target_ifid")

    print(IS_dict)
    print(FID_dict)
    print(iFID_remain_dict)
    print(iFID_target_dict)

 
    # max_IS
    # min_FID
    # min_iFID_remain
    # MIN_IFID

    
    # print("{i}th model IS convergence index: {iter} iterations".format(i="naive", iter = save_freq * IS_list.index(max_IS)))
    # print("{i}th model FID convergence index: {iter} iterations".format(i="naive", iter = save_freq * FID_list.index(min_FID)))
    # print("{i}th model intra_fid_remain convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_remain_list.index(min_iFID_remain)))
    # print("{i}th model intra_fid_target convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_target_list.index(min_iFID_target)))


for i in range(1):
    metrics = np.load("./results/final/{folder}/{data}/statistics/{topg_100_path}/train/metrics.npy".format(folder=fine_folder, data=data, topg_100_path=topg_100_path), allow_pickle=True)
    iFID_remain = np.load("./results/final/{folder}/{data}/statistics/{topg_100_path}/iFID_remain.npy".format(folder=fine_folder, data=data, topg_100_path=topg_100_path), allow_pickle=True)
    iFID_target = np.load("./results/final/{folder}/{data}/statistics/{topg_100_path}/iFID_target.npy".format(folder=fine_folder, data=data, topg_100_path=topg_100_path), allow_pickle=True)

    methods = "topg_100"

    IS_dict[methods] = metrics.item().get("IS")
    FID_dict[methods] = metrics.item().get("FID")
    iFID_remain_dict[methods] = iFID_remain.item().get("remain_ifid")
    iFID_target_dict[methods] = iFID_target.item().get("target_ifid")

    print(IS_dict)
    print(FID_dict)
    print(iFID_remain_dict)
    print(iFID_target_dict)

 
    # max_IS
    # min_FID
    # min_iFID_remain
    # MIN_IFID

    
    # print("{i}th model IS convergence index: {iter} iterations".format(i="naive", iter = save_freq * IS_list.index(max_IS)))
    # print("{i}th model FID convergence index: {iter} iterations".format(i="naive", iter = save_freq * FID_list.index(min_FID)))
    # print("{i}th model intra_fid_remain convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_remain_list.index(min_iFID_remain)))
    # print("{i}th model intra_fid_target convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_target_list.index(min_iFID_target)))


for i in range(1):
    metrics = np.load("./results/final/{folder}/{data}/statistics/{topg_200_path}/train/metrics.npy".format(folder=fine_folder, data=data, topg_200_path=topg_200_path), allow_pickle=True)
    iFID_remain = np.load("./results/final/{folder}/{data}/statistics/{topg_200_path}/iFID_remain.npy".format(folder=fine_folder, data=data, topg_200_path=topg_200_path), allow_pickle=True)
    iFID_target = np.load("./results/final/{folder}/{data}/statistics/{topg_200_path}/iFID_target.npy".format(folder=fine_folder, data=data, topg_200_path=topg_200_path), allow_pickle=True)

    methods = "topg_200"

    IS_dict[methods] = metrics.item().get("IS")
    FID_dict[methods] = metrics.item().get("FID")
    iFID_remain_dict[methods] = iFID_remain.item().get("remain_ifid")
    iFID_target_dict[methods] = iFID_target.item().get("target_ifid")

    print(IS_dict)
    print(FID_dict)
    print(iFID_remain_dict)
    print(iFID_target_dict)

 
    # max_IS
    # min_FID
    # min_iFID_remain
    # MIN_IFID

    
    # print("{i}th model IS convergence index: {iter} iterations".format(i="naive", iter = save_freq * IS_list.index(max_IS)))
    # print("{i}th model FID convergence index: {iter} iterations".format(i="naive", iter = save_freq * FID_list.index(min_FID)))
    # print("{i}th model intra_fid_remain convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_remain_list.index(min_iFID_remain)))
    # print("{i}th model intra_fid_target convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_target_list.index(min_iFID_target)))



for i in range(1):
    metrics = np.load("./results/final/{folder}/{data}/statistics/{topg_1000_path}/train/metrics.npy".format(folder=fine_folder, data=data, topg_1000_path=topg_1000_path), allow_pickle=True)
    iFID_remain = np.load("./results/final/{folder}/{data}/statistics/{topg_1000_path}/iFID_remain.npy".format(folder=fine_folder, data=data, topg_1000_path=topg_1000_path), allow_pickle=True)
    iFID_target = np.load("./results/final/{folder}/{data}/statistics/{topg_1000_path}/iFID_target.npy".format(folder=fine_folder, data=data, topg_1000_path=topg_1000_path), allow_pickle=True)

    methods = "topg_1000"

    IS_dict[methods] = metrics.item().get("IS")
    FID_dict[methods] = metrics.item().get("FID")
    iFID_remain_dict[methods] = iFID_remain.item().get("remain_ifid")
    iFID_target_dict[methods] = iFID_target.item().get("target_ifid")

    print(IS_dict)
    print(FID_dict)
    print(iFID_remain_dict)
    print(iFID_target_dict)

 
    # max_IS
    # min_FID
    # min_iFID_remain
    # MIN_IFID

    
    # print("{i}th model IS convergence index: {iter} iterations".format(i="naive", iter = save_freq * IS_list.index(max_IS)))
    # print("{i}th model FID convergence index: {iter} iterations".format(i="naive", iter = save_freq * FID_list.index(min_FID)))
    # print("{i}th model intra_fid_remain convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_remain_list.index(min_iFID_remain)))
    # print("{i}th model intra_fid_target convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_target_list.index(min_iFID_target)))



x=[]
x.append(0)

x_len = len(IS_dict["topg_2"])
# create data
for i in range(x_len):
    # x.append(save_freq * (i+1))
    x.append('{num}K'.format(num=i+1))

y_top2 = iFID_target_dict["topg_2"]
# y_top5 = iFID_target_dict["topg_5"]
y_top10 = iFID_target_dict["topg_10"]
y_top100 = iFID_target_dict["topg_100"]
# y_top200 = iFID_target_dict["topg_200"]
y_top1000 = iFID_target_dict["topg_1000"]

y_top2.insert(0, pre_train_target_iFID)
# y_top5.insert(0, pre_train_target_iFID)
y_top10.insert(0, pre_train_target_iFID)
y_top100.insert(0, pre_train_target_iFID)
# y_top200.insert(0, pre_train_target_iFID)
y_top1000.insert(0, pre_train_target_iFID)

x = x[:split]
y_top2 = y_top2[:split]
# y_top5 = y_top5[:split]
y_top10 = y_top10[:split]
y_top100 = y_top100[:split]
# y_top200 = y_top200[:split]
y_top1000 = y_top1000[:split]


if data == "FashionMNIST" or data == "CIFAR10":
    x = x[0::2]
    y_top2 = y_top2[:split]
    # y_top5 = y_top5[:split]
    y_top10 = y_top10[:split]
    y_top100 = y_top100[:split]
    # y_top200 = y_top200[:split]
    y_top1000 = y_top1000[:split]

# plot lines
plt.figure(figsize=(6,3))
plt.title("Top % generator weights re-initialization")
plt.plot(x, y_top2, label = "Top 50%", marker ='o', markersize=6, linestyle = '--')
# plt.plot(x, y_top5, label = "Top 20%", marker ='^', markersize=6, linestyle = '--')
plt.plot(x, y_top10, label = "Top 10%", marker ='<', markersize=6, linestyle = '--')
plt.plot(x, y_top100, label = "Top 1%", marker ='s', markersize=6, linestyle = '--')
# plt.plot(x, y_top200, label = "Top 0.5%", marker ='p', markersize=6, linestyle = '--')
plt.plot(x, y_top1000, label = "Top 0.1%", marker ='D', markersize=6, linestyle = '--')

plt.xlabel('# of iterations', fontsize=12)
plt.ylabel('Intra-class FID for the new class', fontsize=12)

# max1 = max(y_ours)
# max2 = max(y_naive)
# max = max(max1,max2)
# plt.ylim(0)

plt.legend()
plt.show()

