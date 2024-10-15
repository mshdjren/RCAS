from torchvision.datasets import ImageFolder

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import random
import torch.backends.cudnn as cudnn
import wandb

import sklearn
from sklearn import metrics
import datetime
import os
import torch.optim as optim


import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics._plot.roc_curve import roc_curve
import tqdm
import torch 
# # pre_train_target_iFID = 172.279
# split = 1+20
# #FOR fASHIONNNIST
# naive_folder = "mas_fine_target_1_last_forget_metric"
# #ELSE
# # naive_folder = "mas_fine_target_1_last"


# fine_folder = "mas_fine_target_1_last_forget_metric"


# # data = "MNIST"
# # naive_path = "MNIST-deep_conv_aux-MNIST_1class_g_lr_0.0001_d_up_2-2023_09_05_01_45_38"
# # ours_path = "MNIST-deep_conv_aux-MNIST_1class_topg_5_topd_200_g_lr_0.0001_d_up_2-2023_09_05_01_26_12"

# # data = "FashionMNIST"
# # naive_path = "FashionMNIST-deep_conv_aux-FashionMNIST_1class_naive_g_lr_0.0001_d_up_2-2023_09_05_01_43_26"
# # ours_path = "FashionMNIST-deep_conv_aux-FashionMNIST_1class_topg_5_topd_200_g_lr_0.0001_d_up_2-2023_09_05_01_24_36"

# data = "CIFAR10"
# naive_path = "CIFAR10-deep_conv_aux-CIFAR10_1class_g_lr_0.0001_d_up_2-2023_09_05_01_46_16"
# ours_path = "CIFAR10-deep_conv_aux-CIFAR10_1class_topg_5_topd_200_g_lr_0.0001_d_up_2-2023_09_05_01_26_34"

import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-mode", type=str, default="Fine-tuning")          # extra value
parser.add_argument("-data", type=str, default="FashionMNIST")          # extra value
parser.add_argument("-name", type=str, default="FashionMNIST_finetuning")          # extra value

args = parser.parse_args()

default = "./data/classifier"

train_path = os.path.join(default, args.data, args.mode, "data")
test_path = os.path.join(default, args.data, "original_data")

if args.data == "MNIST" or args.data == "FashionMNIST":

    train_transform = transforms.Compose([
        transforms.Resize(224, Image.NEAREST),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
else:
    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])])

   

train_data = ImageFolder(root=train_path,
                         transform=train_transform,
                         target_transform=None)

test_data = ImageFolder(root=test_path,
                         transform=train_transform,
                         target_transform=None)

# for i, (data, target) in enumerate(train_data):
#     print(data)
#     print(data.size())

wandb.finish()

wandb.login()

dd = test_data.class_to_idx
ddd = train_data.class_to_idx

print(dd)
print(ddd)

# seed number fix
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = False
cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True, progress = False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)


lr = 1e-3
num_epoch = 100
batch_size = 512

config  = {
    'epoch': num_epoch,
    'batch_size': batch_size,
    'learning_rate': lr,
    'pretrained': "pretrained",
    
} 


# def train_model(x, y, model, criterion, optimizer):
#   # write training code
#   optimizer.zero_grad()
#   pred = model(x)
#   pred = pred.squeeze()
#   # print(pred)
#   # print(y)
#   loss = criterion(pred, y)
# #   loss2 = CB_loss(y, pred, [3187,313], 2, "sigmoid", 0.9999, 2)
#   pred_label = F.softmax(pred)
#   pred_label = torch.round(prob)
#   pred_label = pred_label.detach().cpu().numpy() # pred_label

#   loss.backward()
#   optimizer.step()

#   y = y.detach().cpu().numpy()
#   acc = np.sum(y == pred_label) / y.shape[0]
  
#   return loss2.item(), acc


# def test_model(x, y, model, criterion):
#   with torch.no_grad():
#     model.eval()
#     pred = model(x)
#     pred = pred.squeeze()

#     loss = criterion(pred, y)
#     loss2 = focal_loss(pred, y, 0.75, 0)
#     prob = F.sigmoid(pred)
#     pred_label = torch.round(prob)
#     pred_label = pred_label.detach().cpu().numpy() # pred_label

#     y = y.detach().cpu().numpy()
#     acc = np.sum(y == pred_label) / y.shape[0]

#   return loss.item(), acc, prob.detach().cpu().numpy(), pred_label

def split_array(arr):
    mid = len(arr) // 2
    return arr[:mid], arr[mid:]


wandb.init(project="classifier", config=config, name=args.name)

save_path = os.path.join(train_path,"/results")
os.makedirs(save_path, exist_ok=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
#criterion2 = CB_loss()
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size)

model = model.to(device)
iteration = 0

for epoch in range(num_epoch):
    model.train()
    train_loss, train_acc = 0, 0
    test_loss, test_acc = 0, 0

    gt_list = []
    pred_list = []
    pred_label_list = []

    for x, y in tqdm.tqdm(train_loader, '| train | epoch | %d' % epoch):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        # _, pred = torch.max(model(x),1)
        loss = criterion(pred, y)
        loss.backward()

        # loss, acc = train_model(x, y, model, criterion, optimizer)
        y = y.detach().cpu().numpy()
        _, pred = torch.max(pred,1)
        pred = pred.detach().cpu().numpy() # pred_label
        acc = np.sum(y == pred) / y.shape[0]
        train_loss += loss
        train_acc += acc 

        print('\n', 'iteration %d |' % iteration, f'loss: {loss:.5f}, acc: {acc:.5f}')
        iteration += 1

    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)

    for x, y in tqdm.tqdm(test_loader, '| test | epoch | %d' % epoch):
        x = x.to(device)
        y = y.to(device)
        y = y.type(torch.float)

        # loss, acc = train_model(x, y, model, criterion, optimizer)
        with torch.no_grad():
            model.eval()
            y = y.detach().cpu().numpy()
            pred = model(x)
            _,pred_label = torch.max(pred,1)
            pred_label = pred_label.detach().cpu().numpy() # pred_label
            acc = np.sum(y == pred_label) / y.shape[0]

            # test_loss += loss
            test_acc += acc


            # gt_list.append(y.detach().cpu().numpy())
            # pred_list.append(pred)
            # pred_label_list.append(pred_label)

    # test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)

    print('\n', 'epoch %d |' % epoch, f' test_acc: {test_acc:.5f}')

    wandb.log({
        "train_loss": train_loss,
        # "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
    })

    if epoch == 0:
        min_acc = test_acc
    elif test_acc > min_acc:
        print(f'test_acc has been improved from {min_acc:.5f} to {test_acc:.5f}. Saving Model!')
        # min_loss = test_loss
        file_name = os.path.join(save_path, 'classifier.pth')
        torch.save(model.state_dict(), file_name)

wandb.finish()
