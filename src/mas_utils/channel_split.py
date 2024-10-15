import random
import torch
from utils.misc import peel_model

def Mas():
    reg_params={}
    reg_params_intersect={}

    return reg_params, reg_params_intersect

def init_reg_params(model, device, selected_blocks={}, selected_layers = [], linear1 = False, linear2 = False, last_conv = False, dis_change = False):
    
    model = peel_model(model)
    reg_params, reg_params_intersect = Mas()
    block_names = []

    for name, param in model.named_parameters():
        block_names.append(name)

    for name, param in model.named_parameters():
        strings = name.split('.')

        if strings[0] == 'blocks':
            if int(strings[1]) in selected_blocks:
                if "conv" in strings[3]:
                    if strings[4] == "weight" or strings[4] == "weight_orig":
                        for num in selected_layers:
                            block_names.remove(name)

        if strings[0] == "linear1" or strings[0] == "linear2":
            if linear1 == True:
                if strings[1] == "weight":
                    block_names.remove(name)
            elif linear2 == True:
                if strings[1] == "weight":
                    block_names.remove(name)
            else:
                pass

        if strings[0] == "conv1":
            if last_conv == True:
                if strings[1] == "weight":
                    block_names.remove(name)

        if "deconv" in strings[0]:
            if 99 in selected_layers:
                if strings[1] == "weight" or "weight_orig":
                    block_names.remove(name)

    for name, param in model.named_parameters():
        freeze = False
        for block_name in block_names:
            if block_name in name:
                freeze = True

        if freeze == False:
            print("Initializing omega values for layer", name)
            print("omega.size", param.size())
            if dis_change == False:
                omega = torch.zeros(param.size(dim=1))
            else:
                omega = torch.zeros(param.size(dim=0))
            
            omega = omega.to(device)
            reg_params[name] = omega
            reg_params_intersect[name] = omega

    return block_names, reg_params, reg_params_intersect
