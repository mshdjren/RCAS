# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/ema.py

import random

import torch

from utils.misc import peel_model


def Mas():
    reg_params = {}
    reg_params_intersect = {}

    return reg_params, reg_params_intersect

def init_reg_params(model, device, selected_blocks = [], selected_layers = [], linear1 = False, linear2 = False, last_conv = False, dis_change = False):

	model = peel_model(model)
	reg_params, reg_params_intersect = Mas()
	block_names=[]
	"""
	Input:
	1) model: A reference to the model that is being trained
	2) use_gpu: Set the flag to True if the model is to be trained on the GPU
	3) freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
		case of computational limitations where computing the importance parameters for the entire model
		is not feasible
	Output:
	1) model: A dictionary containing importance weights (omega), init_val (keep a reference 
	to the initial values of the parameters) for all trainable parameters is calculated and the updated
	model with these reg_params is returned.
	Function: Initializes the reg_params for a model for the initial task (task = 1)	

	"""
	# for layer in range(num_freeze_layers):
	# 	block_names.append("blocks.{layer}".format(layer=layer))
	"set freezeing layers"
	for name, param in model.named_parameters():
		block_names.append(name)
	
	for name, param in model.named_parameters():
		strings = name.split('.')

		if strings[0] == 'blocks':
			if int(strings[1]) in selected_blocks:
				if "conv" in strings[3]:
					if strings[4] == "weight" or strings[4] == "weight_orig":
						# for num in selected_layers:
						if int(strings[3][-1]) in selected_layers:
							block_names.remove(name)

		if strings[0] == 'linear1' or strings[0] == 'linear2':
			if linear1 == True:
				if strings[1] == "weight" or strings[1] == "weight_orig":
					block_names.remove(name)
			elif linear2 == True:
				if strings[1] == "weight" or strings[1] == "weight_orig":
					block_names.remove(name)
			else:
				pass
					
		if strings[0] == 'conv1':
			if last_conv == True:
				if strings[1] == "weight" or strings[1] == "weight_orig":
					block_names.remove(name)
					
		# if "deconv" in strings[0]:
		# 	if 99 in selected_layers:
		# 		if strings[1] == "weight":
		# 		# "for BigGAN"
		# 		# if strings[1] == "weight_orig":
		# 			block_names.remove(name)


	for name, param in model.named_parameters():
		freeze = False
		for block_name in block_names:
			if block_name == name:
				freeze = True
		if freeze == False:
			print ("Initializing omega values for layer", name)
			omega = torch.zeros(param.size())
			omega = omega.to(device)

			init_val = param.data.clone()
			param_dict = {}

			#for first task, omega is initialized to zero
			param_dict['omega'] = omega
			param_dict['init_val'] = init_val

			#the key for this dictionary is the name of the layer
			reg_params[param] = param_dict
			reg_params_intersect[param] = param_dict


	return block_names, reg_params, reg_params_intersect