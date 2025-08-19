#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:01:07 2025

@author: arshdeep
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:42:47 2025

@author: arshdeep
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torchinfo import summary
import random
from pruned_algo import operator_norm_pruning, ICLR_L1_Imp_index, ICLR_GM_Imp_index
import numpy as np
import pickle

# %%


def channel_wise_score(key, w_org):
    weights = w_org
    z = key.split('.')
    W_2D = weights[key]
    if ('pwconv1' in z) or ('pwconv2' in z):
        W_2D = W_2D.unsqueeze(2).unsqueeze(3)
    W_2D = W_2D.numpy()
    W = np.reshape(W_2D, (np.shape(W_2D)[0], np.shape(W_2D)[
                   1], np.shape(W_2D)[2]*np.shape(W_2D)[3]))
    print(W.shape)  # W Shape: [FILTERS, CHANNELS, HEIGHT, WIDTH]
    # change function here to generate sorted index
    score = operator_norm_pruning(W)
    # a high score means high importance, sort the score indexex
    return np.array(score)


# %% load original weights

# original weights
state_dict_path = '/home/arshdeep/ConvNeXt/checkpoints/convnext_tiny_1k_224_ema.pth'
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))

w_org = state_dict['model']
# %%
'''
list_pruned_layers = ['downsample_layers.3.1.weight', 'stages.3.0.dwconv.weight', 'stages.3.0.pwconv1.weight', 'stages.3.0.pwconv2.weight', 'stages.3.1.dwconv.weight',
                      'stages.3.1.pwconv1.weight', 'stages.3.1.pwconv2.weight', 'stages.3.2.dwconv.weight', 'stages.3.2.pwconv1.weight', 'stages.3.2.pwconv2.weight', '']
# tensor_reshaped  = Q.view(3072, 768, 1, 1) #Q.unsqueeze(2).unsqueeze(3)


High_key = [   'stages.0.0.dwconv.weight', 'stages.0.0.pwconv1.weight', 'stages.0.0.pwconv2.weight', 
            'stages.0.1.dwconv.weight', 'stages.0.1.pwconv1.weight', 'stages.0.1.pwconv2.weight', 
            'stages.0.2.dwconv.weight', 'stages.0.2.pwconv1.weight', 'stages.0.2.pwconv2.weight',
           'stages.1.0.dwconv.weight', 'stages.1.0.pwconv1.weight', 'stages.1.0.pwconv2.weight', 
           'stages.1.1.dwconv.weight', 'stages.1.1.pwconv1.weight','stages.1.1.pwconv2.weight',
           'stages.1.2.dwconv.weight', 'stages.1.2.pwconv1.weight', 'stages.1.2.pwconv2.weight',
           'stages.2.0.dwconv.weight', 'stages.2.0.pwconv1.weight', 'stages.2.0.pwconv2.weight', 
            'stages.2.1.dwconv.weight', 'stages.2.1.pwconv1.weight','stages.2.1.pwconv2.weight', 
              'stages.2.2.dwconv.weight', 'stages.2.2.pwconv1.weight', 'stages.2.2.pwconv2.weight',
              'stages.2.3.dwconv.weight', 'stages.2.3.pwconv1.weight', 'stages.2.3.pwconv2.weight',
               'stages.2.4.dwconv.weight', 'stages.2.4.pwconv1.weight','stages.2.4.pwconv2.weight', 
                'stages.2.5.dwconv.weight', 'stages.2.5.pwconv1.weight', 'stages.2.5.pwconv2.weight',
                  'stages.2.6.dwconv.weight', 'stages.2.6.pwconv1.weight', 'stages.2.6.pwconv2.weight', 
                    'stages.2.7.dwconv.weight', 'stages.2.7.pwconv1.weight', 'stages.2.7.pwconv2.weight', 
                      'stages.2.8.dwconv.weight', 'stages.2.8.pwconv1.weight', 'stages.2.8.pwconv2.weight',
                        'stages.2.8.dwconv.weight', 'stages.2.8.pwconv1.weight', 'stages.2.8.pwconv2.weight',
                         'stages.3.0.dwconv.weight', 'stages.3.0.pwconv1.weight', 'stages.3.0.pwconv2.weight', 
                          'stages.3.1.dwconv.weight', 'stages.3.1.pwconv1.weight', 'stages.3.1.pwconv2.weight', 
                           'stages.3.2.dwconv.weight', 'stages.3.2.pwconv1.weight', 'stages.3.2.pwconv2.weight', 
                           'downsample_layers.0.0.weight','downsample_layers.1.1.weight','downsample_layers.2.2.weight', 'downsample_layers.3.3.weight']


    

# High_key = ['downsample_layers.3.1.weight']

sorted_index_dict = {}

for key in High_key:
    score = channel_wise_score(key, w_org)
    # low to high score, high is more important
    sorted_index = np.argsort(score)
    sorted_index_dict[key] = sorted_index

# %% assign sorted indeexes to connected layers....


key_dwconv_stage3_0 = ['stages.3.0.dwconv.bias', 'stages.3.0.dwconv.gamma',
                       'stages.3.0.norm.bias', 'stages.3.0.norm.weight']

key_dwconv_stage3_1 = ['stages.3.1.dwconv.bias', 'stages.3.1.dwconv.gamma',
                       'stages.3.1.norm.bias', 'stages.3.1.norm.weight']

key_dwconv_stage3_2 = ['stages.3.2.dwconv.bias', 'stages.3.2.dwconv.gamma',
                       'stages.3.2.norm.bias', 'stages.3.2.norm.weight']


for key in key_dwconv_stage3_0:
    sorted_index_dict[key] = sorted_index_dict['stages.3.0.dwconv.weight']

for key in key_dwconv_stage3_1:
    sorted_index_dict[key] = sorted_index_dict['stages.3.1.dwconv.weight']

for key in key_dwconv_stage3_2:
    sorted_index_dict[key] = sorted_index_dict['stages.3.2.dwconv.weight']

# pwconv1 layer
sorted_index_dict['stages.3.0.pwconv1.bias'] = sorted_index_dict['stages.3.0.pwconv1.weight']
sorted_index_dict['stages.3.1.pwconv1.bias'] = sorted_index_dict['stages.3.1.pwconv1.weight']
sorted_index_dict['stages.3.2.pwconv1.bias'] = sorted_index_dict['stages.3.2.pwconv1.weight']


# pwconv2 layer
sorted_index_dict['stages.3.0.pwconv2.bias'] = sorted_index_dict['stages.3.0.pwconv2.weight']
sorted_index_dict['stages.3.1.pwconv2.bias'] = sorted_index_dict['stages.3.1.pwconv2.weight']
sorted_index_dict['stages.3.2.pwconv2.bias'] = sorted_index_dict['stages.3.2.pwconv2.weight']

# downsample layer
sorted_index_dict['downsample_layers.3.1.bias'] = sorted_index_dict['downsample_layers.3.1.weight']


# gamma
sorted_index_dict['stages.3.0.gamma'] = sorted_index_dict['stages.3.0.pwconv2.weight']
sorted_index_dict['stages.3.1.gamma'] = sorted_index_dict['stages.3.1.pwconv2.weight']
sorted_index_dict['stages.3.2.gamma'] = sorted_index_dict['stages.3.2.pwconv2.weight']

# norm after all stages (3)
sorted_index_dict['norm.weight'] = sorted_index_dict['stages.3.2.pwconv2.weight']
sorted_index_dict['norm.bias'] = sorted_index_dict['stages.3.2.pwconv2.weight']
# %%



# %%

keys_to_pruned = ['downsample_layers.3.1.weight', 'downsample_layers.3.1.bias','stages.3.0.gamma',
 'stages.3.0.dwconv.weight',
 'stages.3.0.dwconv.bias',
 'stages.3.0.norm.weight',
 'stages.3.0.norm.bias',
 'stages.3.0.pwconv1.weight',
 'stages.3.0.pwconv1.bias',
 'stages.3.0.pwconv2.weight',
 'stages.3.0.pwconv2.bias',
 'stages.3.1.gamma',
 'stages.3.1.dwconv.weight',
 'stages.3.1.dwconv.bias',
 'stages.3.1.norm.weight',
 'stages.3.1.norm.bias',
 'stages.3.1.pwconv1.weight',
 'stages.3.1.pwconv1.bias',
 'stages.3.1.pwconv2.weight',
 'stages.3.1.pwconv2.bias',
 'stages.3.2.gamma',
 'stages.3.2.dwconv.weight',
 'stages.3.2.dwconv.bias',
 'stages.3.2.norm.weight',
 'stages.3.2.norm.bias',
 'stages.3.2.pwconv1.weight',
 'stages.3.2.pwconv1.bias',
 'stages.3.2.pwconv2.weight',
 'stages.3.2.pwconv2.bias',
 'norm.weight',
 'norm.bias']
'''
#%%

High_key = [
    # Stage 0
    'stages.0.0.dwconv.weight', 'stages.0.0.pwconv1.weight', 'stages.0.0.pwconv2.weight',
    'stages.0.1.dwconv.weight', 'stages.0.1.pwconv1.weight', 'stages.0.1.pwconv2.weight',
    'stages.0.2.dwconv.weight', 'stages.0.2.pwconv1.weight', 'stages.0.2.pwconv2.weight',
    
    # Stage 1
    'stages.1.0.dwconv.weight', 'stages.1.0.pwconv1.weight', 'stages.1.0.pwconv2.weight',
    'stages.1.1.dwconv.weight', 'stages.1.1.pwconv1.weight', 'stages.1.1.pwconv2.weight',
    'stages.1.2.dwconv.weight', 'stages.1.2.pwconv1.weight', 'stages.1.2.pwconv2.weight',
    
    # Stage 2 (depth 9)
    'stages.2.0.dwconv.weight', 'stages.2.0.pwconv1.weight', 'stages.2.0.pwconv2.weight',
    'stages.2.1.dwconv.weight', 'stages.2.1.pwconv1.weight', 'stages.2.1.pwconv2.weight',
    'stages.2.2.dwconv.weight', 'stages.2.2.pwconv1.weight', 'stages.2.2.pwconv2.weight',
    'stages.2.3.dwconv.weight', 'stages.2.3.pwconv1.weight', 'stages.2.3.pwconv2.weight',
    'stages.2.4.dwconv.weight', 'stages.2.4.pwconv1.weight', 'stages.2.4.pwconv2.weight',
    'stages.2.5.dwconv.weight', 'stages.2.5.pwconv1.weight', 'stages.2.5.pwconv2.weight',
    'stages.2.6.dwconv.weight', 'stages.2.6.pwconv1.weight', 'stages.2.6.pwconv2.weight',
    'stages.2.7.dwconv.weight', 'stages.2.7.pwconv1.weight', 'stages.2.7.pwconv2.weight',
    'stages.2.8.dwconv.weight', 'stages.2.8.pwconv1.weight', 'stages.2.8.pwconv2.weight',
    
    # Stage 3 (depth 3)
    'stages.3.0.dwconv.weight', 'stages.3.0.pwconv1.weight', 'stages.3.0.pwconv2.weight',
    'stages.3.1.dwconv.weight', 'stages.3.1.pwconv1.weight', 'stages.3.1.pwconv2.weight',
    'stages.3.2.dwconv.weight', 'stages.3.2.pwconv1.weight', 'stages.3.2.pwconv2.weight',
    
    # downsampling layers
    'downsample_layers.0.0.weight','downsample_layers.1.1.weight','downsample_layers.2.1.weight', 'downsample_layers.3.1.weight'
]

sorted_index_dict = {}

for key in High_key:
    score = channel_wise_score(key, w_org)
    sorted_index = np.argsort(score)
    sorted_index_dict[key] = sorted_index

print('length of sorted index', len(sorted_index_dict))

# Assign sorted indexes to connected layers
for stage in range(4):
    depth = [3, 3, 9, 3][stage]  # Depths per stage
    for d in range(depth):
        key_dwconv = [
            f'stages.{stage}.{d}.dwconv.bias',
            f'stages.{stage}.{d}.norm.bias', f'stages.{stage}.{d}.norm.weight'
        ]
        print(key_dwconv)
        for key in key_dwconv:
            sorted_index_dict[key] = sorted_index_dict[f'stages.{stage}.{d}.dwconv.weight']
        
        # pwconv1 and pwconv2
        sorted_index_dict[f'stages.{stage}.{d}.pwconv1.bias'] = sorted_index_dict[f'stages.{stage}.{d}.pwconv1.weight']
        sorted_index_dict[f'stages.{stage}.{d}.pwconv2.bias'] = sorted_index_dict[f'stages.{stage}.{d}.pwconv2.weight']
        
        # Gamma
        sorted_index_dict[f'stages.{stage}.{d}.gamma'] = sorted_index_dict[f'stages.{stage}.{d}.pwconv2.weight']


# downsample layer

sorted_index_dict['downsample_layers.0.0.bias'] = sorted_index_dict['downsample_layers.0.0.weight']
sorted_index_dict['downsample_layers.0.1.weight'] = sorted_index_dict['downsample_layers.0.0.weight']
sorted_index_dict['downsample_layers.0.1.bias'] = sorted_index_dict['downsample_layers.0.0.weight']



sorted_index_dict['downsample_layers.1.0.weight'] = sorted_index_dict['stages.0.2.pwconv2.weight']
sorted_index_dict['downsample_layers.1.0.bias'] = sorted_index_dict['stages.0.2.pwconv2.weight']
sorted_index_dict['downsample_layers.1.1.bias'] = sorted_index_dict['downsample_layers.1.1.weight']


sorted_index_dict['downsample_layers.2.0.weight'] = sorted_index_dict['stages.1.2.pwconv2.weight']
sorted_index_dict['downsample_layers.2.0.bias'] =  sorted_index_dict['stages.1.2.pwconv2.weight']
sorted_index_dict['downsample_layers.2.1.bias'] = sorted_index_dict['downsample_layers.2.1.weight']




sorted_index_dict['downsample_layers.3.0.weight'] =  sorted_index_dict['stages.2.8.pwconv2.weight']
sorted_index_dict['downsample_layers.3.0.bias'] = sorted_index_dict['stages.2.8.pwconv2.weight']
sorted_index_dict['downsample_layers.3.1.bias'] = sorted_index_dict['downsample_layers.3.1.weight']




# norm layer penultimate
sorted_index_dict['norm.weight'] = sorted_index_dict['stages.3.2.pwconv2.weight']
sorted_index_dict['norm.bias'] = sorted_index_dict['stages.3.2.pwconv2.weight']
sorted_index_dict['head.weight'] = sorted_index_dict['stages.3.2.pwconv2.weight']
sorted_index_dict['head.bias'] = sorted_index_dict['stages.3.2.pwconv2.weight']



#%%

os.chdir('/home/arshdeep/ConvNeXt/sorted_index_imagenet_alllayers/OP/')

# Sample dictionary
# data = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# Save dictionary to a file
with open('sorted_index_convnext_op.pkl', 'wb') as f:
    pickle.dump(sorted_index_dict, f)
    