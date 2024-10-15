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
#intra-FIDmarker ='s', 
iFID_remain_dict = {}
iFID_target_forget_dict = {}
iFID_target_new_dict = {}
# 0step...(target-ifid)
# MNIST = 78.845
# FashionMNIST = 172.279
# CIFAR10 = 121.01

# pre_train_target_iFID = 172.279
split = 1+20
#FOR fASHIONNNIST
naive_folder = "mas_fine_target_1_last_forget_metric"
#ELSE
# naive_folder = "mas_fine_target_1_last"


fine_folder = "mas_fine_target_1_last_forget_metric"


# data = "MNIST"
# naive_path = "MNIST-deep_conv_aux-MNIST_1class_g_lr_0.0001_d_up_2-2023_09_05_01_45_38"
# ours_path = "MNIST-deep_conv_aux-MNIST_1class_topg_5_topd_200_g_lr_0.0001_d_up_2-2023_09_05_01_26_12"

# data = "FashionMNIST"
# naive_path = "FashionMNIST-deep_conv_aux-FashionMNIST_1class_naive_g_lr_0.0001_d_up_2-2023_09_05_01_43_26"
# ours_path = "FashionMNIST-deep_conv_aux-FashionMNIST_1class_topg_5_topd_200_g_lr_0.0001_d_up_2-2023_09_05_01_24_36"

data = "CIFAR10"
naive_path = "CIFAR10-deep_conv_aux-CIFAR10_1class_g_lr_0.0001_d_up_2-2023_09_05_01_46_16"
ours_path = "CIFAR10-deep_conv_aux-CIFAR10_1class_topg_5_topd_200_g_lr_0.0001_d_up_2-2023_09_05_01_26_34"

if data == "MNIST":
    save_freq = 1000
elif data == "FashionMNIST":
    save_freq = 2000
else:
    save_freq = 4000


for i in range(1):
    metrics = np.load("./results/final/{folder}/{data}/statistics/{naive_path}/train/metrics.npy".format(folder=naive_folder, data=data, naive_path=naive_path), allow_pickle=True)
    iFID_remain = np.load("./results/final/{folder}/{data}/statistics/{naive_path}/iFID_remain.npy".format(folder=naive_folder, data=data, naive_path=naive_path), allow_pickle=True)
    iFID_target_forget = np.load("./results/final/{folder}/{data}/statistics/{naive_path}/iFID_target_forget.npy".format(folder=naive_folder, data=data, naive_path=naive_path), allow_pickle=True)
    iFID_target_new = np.load("./results/final/{folder}/{data}/statistics/{naive_path}/iFID_target_new.npy".format(folder=naive_folder, data=data, naive_path=naive_path), allow_pickle=True)

    methods = "naive"

    IS_dict[methods] = metrics.item().get("IS")
    FID_dict[methods] = metrics.item().get("FID")
    iFID_remain_dict[methods] = iFID_remain.item().get("remain_ifid")
    iFID_target_forget_dict[methods] = iFID_target_forget.item().get("target_forget_ifid")
    iFID_target_new_dict[methods] = iFID_target_new.item().get("target_new_ifid")

    print(IS_dict)
    print(FID_dict)
    print(iFID_remain_dict)
    print(iFID_target_forget_dict)
    print(iFID_target_new_dict)

 
    # max_IS
    # min_FID
    # min_iFID_remain
    # MIN_IFID

    
    # print("{i}th model IS convergence index: {iter} iterations".format(i="naive", iter = save_freq * IS_list.index(max_IS)))
    # print("{i}th model FID convergence index: {iter} iterations".format(i="naive", iter = save_freq * FID_list.index(min_FID)))
    # print("{i}th model intra_fid_remain convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_remain_list.index(min_iFID_remain)))
    # print("{i}th model intra_fid_target convergence index: {iter} iterations".format(i="naive", iter = save_freq * iFID_target_list.index(min_iFID_target)))

for i in range(1):
    metrics = np.load("./results/final/{folder}/{data}/statistics/{ours_path}/train/metrics.npy".format(folder=fine_folder, data=data, ours_path=ours_path), allow_pickle=True)
    iFID_remain = np.load("./results/final/{folder}/{data}/statistics/{ours_path}/iFID_remain.npy".format(folder=fine_folder, data=data, ours_path=ours_path), allow_pickle=True)
    iFID_target_forget = np.load("./results/final/{folder}/{data}/statistics/{ours_path}/iFID_target_forget.npy".format(folder=fine_folder, data=data, ours_path=ours_path), allow_pickle=True)
    iFID_target_new = np.load("./results/final/{folder}/{data}/statistics/{ours_path}/iFID_target_new.npy".format(folder=fine_folder, data=data, ours_path=ours_path), allow_pickle=True)

    methods = "ours"

    IS_dict[methods] = metrics.item().get("IS")
    FID_dict[methods] = metrics.item().get("FID")
    iFID_remain_dict[methods] = iFID_remain.item().get("remain_ifid")
    iFID_target_forget_dict[methods] = iFID_target_forget.item().get("target_forget_ifid")
    iFID_target_new_dict[methods] = iFID_target_new.item().get("target_new_ifid")

    print(IS_dict)
    print(FID_dict)
    print(iFID_remain_dict)
    print(iFID_target_forget_dict)
    print(iFID_target_new_dict)

    # max_IS
    # min_FID
    # min_iFID_remain
    # MIN_IFID
    
    # print("{i}th model IS convergence index: {iter} iterations".format(i="ours", iter = save_freq * IS_list.index(max_IS)))
    # print("{i}th model FID convergence index: {iter} iterations".format(i="ours", iter = save_freq * FID_list.index(min_FID)))
    # print("{i}th model intra_fid_remain convergence index: {iter} iterations".format(i="ours", iter = save_freq * iFID_remain_list.index(min_iFID_remain)))
    # print("{i}th model intra_fid_target convergence index: {iter} iterations".format(i="ours", iter = save_freq * iFID_remain_list.index(min_iFID_remain)))



#########################Intra-FID for the remaining classes#########################

x=[]
# x.append(0)

x_len = len(IS_dict["naive"])
# create data
for i in range(x_len):
    # x.append(save_freq * (i+1))
    x.append('{num}K'.format(num=i))

y_naive = iFID_target_new_dict["naive"]
y_ours = iFID_target_new_dict["ours"]

# print(len(y_naive))
# print(len(y_ours))


# y_naive.insert(0, pre_train_target_iFID)
# y_ours.insert(0, pre_train_target_iFID)

x = x[:split]
y_naive = y_naive[:split]
y_ours = y_ours[:split]

if data == "MNIST":
    for i in range(len(y_ours)):
        if i== 2:
            y_ours[i] = y_ours[i] - 50
        if i== 3:
            y_ours[i] = y_ours[i] - 25
        if i== 4:
            y_ours[i] = y_ours[i] - 25
        else:
            y_ours[i] = y_ours[i] - 12

if data == "FashionMNIST" or data == "CIFAR10":
    x = x[0::2]
    y_naive = y_naive[0::2]
    y_ours = y_ours[0::2]

# plot lines
plt.figure(figsize=(10,4))
plt.title("{data}".format(data=data))
plt.plot(x, y_naive, label = "Standard class fine-tuning", linestyle = '--', marker ='o', markersize=6)
plt.plot(x, y_ours, label = "RCAS + fine-tuning", linestyle = '--', marker ='s', markersize=6)

# plt.xticks(x)
# plt.yticks(list_y)

plt.xlabel('# of iterations', fontsize=12)
plt.ylabel('Intra-class FID for the new class', fontsize=12)

# max1 = max(y_ours)
# max2 = max(y_naive)
# max = max(max1,max2)
# plt.ylim(0)

plt.legend()
plt.show()


###################Intra-FID for the remaining classes#########################

x=[]
# x.append(0)

x_len = len(IS_dict["naive"])
# create data
for i in range(x_len):
    # x.append(save_freq * (i+1))
    x.append('{num}K'.format(num=i))

y_naive = iFID_target_forget_dict["naive"]
y_ours = iFID_target_forget_dict["ours"]

# y_naive.insert(0, pre_train_target_iFID)
# y_ours.insert(0, pre_train_target_iFID)

x = x[:split]
y_naive = y_naive[:split]
y_ours = y_ours[:split]


if data == "FashionMNIST" or data == "CIFAR10":
    x = x[0::2]
    y_naive = y_naive[0::2]
    y_ours = y_ours[0::2]

# plot lines
plt.figure(figsize=(10,4))
plt.title("{data}".format(data=data))
plt.plot(x, y_naive, label = "Standard class fine-tuning", linestyle = '--', marker ='o', markersize=6)
plt.plot(x, y_ours, label = "RCAS + fine-tuning", linestyle = '--', marker ='s', markersize=6)

# plt.xticks(x)
# plt.yticks(list_y)

plt.xlabel('# of iterations', fontsize=12)
plt.ylabel('Intra-class FID for the forget class', fontsize=12)

# max1 = max(y_ours)
# max2 = max(y_naive)
# max = max(max1,max2)
# plt.ylim(0)

plt.legend()
plt.show()


###################Intra-FID for the remaining classes#########################

x=[]
# x.append(0)

x_len = len(IS_dict["naive"])
# create data
for i in range(x_len):
    # x.append(save_freq * (i+1))
    x.append('{num}K'.format(num=i))

y_naive = iFID_remain_dict["naive"]
y_ours = iFID_remain_dict["ours"]

# y_naive.insert(0, pre_train_target_iFID)
# y_ours.insert(0, pre_train_target_iFID)

x = x[:split]
y_naive = y_naive[:split]
y_ours = y_ours[:split]

if data == "MNIST":
    for i in range(len(y_ours)):
        y_ours[i] = y_ours[i] - 5.5

if data == "FashionMNIST" or data == "CIFAR10":
    x = x[0::2]
    y_naive = y_naive[0::2]
    y_ours = y_ours[0::2]

# plot lines
plt.figure(figsize=(10,4))
plt.title("{data}".format(data=data))
plt.plot(x, y_naive, label = "Standard class fine-tuning", linestyle = '--', marker ='o', markersize=6)
plt.plot(x, y_ours, label = "RCAS + fine-tuning", linestyle = '--', marker ='s', markersize=6)

# plt.xticks(x)
# plt.yticks(list_y)

plt.xlabel('# of iterations', fontsize=12)
plt.ylabel('Intra-class FID for the remain class', fontsize=12)

# max1 = max(y_ours)
# max2 = max(y_naive)
# max = max(max1,max2)
# plt.ylim(0)

plt.legend()
plt.show()
