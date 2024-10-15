import os
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms


# def create_dir(dir_name):
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)


# def get_dataset(data_name, path='./data'):
#     if not data_name in ['mnist', 'cifar10']:
#         raise TypeError('data_name should be a string, including mnist,cifar10. ')

#     # model: 2 conv. layers followed by 2 FC layers
#     if (data_name == 'mnist'):
#         trainset = datasets.MNIST(path, train=True, download=True,
#                                   transform=transforms.Compose([
#                                       transforms.ToTensor(),
#                                       transforms.Normalize((0.1307,), (0.3081,))
#                                   ]))
#         testset = datasets.MNIST(path, train=False, download=True,
#                                  transform=transforms.Compose([
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.1307,), (0.3081,))
#                                  ]))

#     # model: ResNet-50
#     elif (data_name == 'cifar10'):
#         transform = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         trainset = datasets.CIFAR10(root=path, train=True,
#                                     download=True, transform=transform)
#         testset = datasets.CIFAR10(root=path, train=False,
#                                    download=True, transform=transform)
#     return trainset, testset


def get_dataloader(trainset, testset, batch_size, device):
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


# def split_class_data(dataset, forget_class, num_forget):
def split_class_data(dataset, target_classes, remain_classes):
    forget_index = []
    # class_remain_index = []
    remain_index = []
    # sum = 0
    for i, (data, target) in enumerate(dataset):
        # if target == forget_class and sum < num_forget:
        if target in target_classes:
            forget_index.append(i)
            # sum += 1
        # elif target == forget_class and sum >= num_forget:
        # elif target == forget_class:
            # class_remain_index.append(i)
            # remain_index.append(i)
            # sum += 1
        elif target in remain_classes:
            remain_index.append(i)
        else:
            pass
    # return forget_index, remain_index, class_remain_index
    return forget_index, remain_index


# def split_dataset(dataset, forget_class):

# def get_unlearn_loader(trainset, testset, forget_class, batch_size, num_forget, repair_num_ratio=0.01):
def get_unlearn_loader(trainset, testset, target_classes, remain_classes, cfgs, logger, load_train_dataset, load_eval_dataset):

    if load_train_dataset:
        train_forget_index, train_remain_index = split_class_data(trainset, target_classes, remain_classes)
        logger.info("Train forget dataset size: {dataset_size}".format(dataset_size=len(train_forget_index)))
        logger.info("Train remain dataset size: {dataset_size}".format(dataset_size=len(train_remain_index)))

        train_forget_sampler = SubsetRandomSampler(train_forget_index)  # 5000
        train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 45000
        
        train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, 
                                                    batch_size=cfgs.OPTIMIZATION.basket_size,
                                                    pin_memory=True,
                                                    num_workers=cfgs.RUN.num_workers,
                                                    sampler=train_forget_sampler,
                                                    drop_last=True)


        # from data_util import Dataset_
        # train_dataset = Dataset_(data_name="FashionMNIST",
        #                     data_dir=cfgs.RUN.data_dir,
        #                     train=True,
        #                     crop_long_edge=cfgs.PRE.crop_long_edge,
        #                     resize_size=cfgs.PRE.resize_size,
        #                     random_flip=cfgs.PRE.apply_rflip,
        #                     normalize=True,
        #                     load_data_in_memory=cfgs.RUN.load_data_in_memory)
        
        # train_dataset = Dataset_(data_name="wafer",
        #             data_dir="./data/wafer_defect",
        #             train=True,
        #             crop_long_edge=cfgs.PRE.crop_long_edge,
        #             resize_size=cfgs.PRE.resize_size,
        #             random_flip=cfgs.PRE.apply_rflip,
        #             normalize=True,
        #             load_data_in_memory=cfgs.RUN.load_data_in_memory)
        
        # train_forget_index = [0]*(len(train_forget_index)-1)

        # train_forget_sampler = SubsetRandomSampler(train_forget_index)  # 5000

        # train_forget_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        #                                     batch_size=cfgs.OPTIMIZATION.basket_size,
        #                                     pin_memory=True,
        #                                     num_workers=cfgs.RUN.num_workers,
        #                                     sampler=train_forget_sampler,
        #                                     drop_last=True)
                
        train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, 
                                                batch_size=cfgs.OPTIMIZATION.basket_size,
                                                pin_memory=True,
                                                num_workers=cfgs.RUN.num_workers,
                                                sampler=train_remain_sampler,
                                                drop_last=True)
    else: 
        train_forget_loader = None
        train_remain_loader = None
        
    if load_eval_dataset:                                                               
        test_forget_index, test_remain_index = split_class_data(testset, target_classes, remain_classes)
        logger.info("test forget dataset size: {dataset_size}".format(dataset_size=len(test_forget_index)))
        logger.info("test remain dataset size: {dataset_size}".format(dataset_size=len(test_remain_index)))

        test_forget_sampler = SubsetRandomSampler(test_forget_index)  # 1000
        test_remain_sampler = SubsetRandomSampler(test_remain_index)  # 9000

       
        test_forget_loader = torch.utils.data.DataLoader(dataset=testset, 
                                                     batch_size=cfgs.OPTIMIZATION.batch_size,
                                                     pin_memory=True,
                                                     num_workers=cfgs.RUN.num_workers,
                                                     sampler=test_forget_sampler,
                                                     drop_last=True)
        
        test_remain_loader = torch.utils.data.DataLoader(dataset=testset, 
                                                     batch_size=cfgs.OPTIMIZATION.batch_size,
                                                     pin_memory=True,
                                                     num_workers=cfgs.RUN.num_workers,
                                                     sampler=test_remain_sampler,
                                                     drop_last=True)
    else: 
        test_forget_loader = None
        test_remain_loader = None

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader


def get_forget_loader(dt, forget_class):
    idx = []
    els_idx = []
    count = 0
    for i in range(len(dt)):
        _, lbl = dt[i]
        if lbl == forget_class:
            # if forget:
            #     count += 1
            #     if count > forget_num:
            #         continue
            idx.append(i)
        else:
            els_idx.append(i)
    forget_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(idx), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(els_idx), drop_last=True)
    return forget_loader, remain_loader
