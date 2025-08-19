
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torchinfo import summary
import random
from pruned_algo import operator_norm_pruning
import numpy as np
import pickle

#%%
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def scale_dims(dims, p):
    return [int(dim * p) for dim in dims]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_tiny_image(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes = 200, **kwargs)
 
    return model


#%%


'''
class ConvNeXt_Pruned(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], p=1, drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()
        pruned_dims = [int(dim * p) for dim in dims]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, pruned_dims[0], kernel_size=4, stride=4),
            LayerNorm(pruned_dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(pruned_dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(pruned_dims[i], pruned_dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=pruned_dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(pruned_dims[-1], eps=1e-6)
        self.head = nn.Linear(pruned_dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def load_pruned_model_weights(original_model, pruned_model):
    original_state_dict = original_model.state_dict()
    pruned_state_dict = pruned_model.state_dict()

    for key in pruned_state_dict.keys():
        original_weight = original_state_dict[key]
        pruned_size = pruned_state_dict[key].shape

        # Prune input and output channels consistently
        if len(original_weight.shape) == 4:  # Conv layers
            out_channels, in_channels, h, w = original_weight.shape
            pruned_out_indices = sorted(random.sample(range(out_channels), pruned_size[0]))
            pruned_in_indices = sorted(random.sample(range(in_channels), pruned_size[1]))
            pruned_weight = original_weight[pruned_out_indices][:, pruned_in_indices]
        elif len(original_weight.shape) == 2:  # Linear layers
            out_features, in_features = original_weight.shape
            pruned_out_indices = sorted(random.sample(range(out_features), pruned_size[0]))
            pruned_in_indices = sorted(random.sample(range(in_features), pruned_size[1]))
            pruned_weight = original_weight[pruned_out_indices][:, pruned_in_indices]
        else:
            pruned_weight = original_weight[:pruned_size[0]]

        pruned_state_dict[key] = pruned_weight

    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model
'''
#%% pruning only in 768 dimensional layer

'''
class ConvNeXt_Pruned(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], p=0.5, drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()
        pruned_dims = [dim if dim != 768 else int(dim * p) for dim in dims]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, pruned_dims[0], kernel_size=4, stride=4),
            LayerNorm(pruned_dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(pruned_dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(pruned_dims[i], pruned_dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=pruned_dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(pruned_dims[-1], eps=1e-6)
        self.head = nn.Linear(pruned_dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def load_pruned_model_weights(original_model, pruned_model):
    original_state_dict = original_model.state_dict()
    pruned_state_dict = pruned_model.state_dict()

    for key in pruned_state_dict.keys():
        if key in original_state_dict:
            original_weight = original_state_dict[key]
            pruned_weight = pruned_state_dict[key]
            if original_weight.shape == pruned_weight.shape:
                pruned_state_dict[key] = original_weight.clone()
            elif original_weight.shape[0] == 768:
                pruned_size_0 = pruned_weight.shape[0]
                pruned_size_1 = pruned_weight.shape[1] if len(original_weight.shape) > 1 else None
                if pruned_size_1:
                    pruned_state_dict[key] = original_weight[:pruned_size_0, :pruned_size_1].clone()
                else:
                    pruned_state_dict[key] = original_weight[:pruned_size_0].clone()

    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model
'''
#%% pruning with slected indexes

class ConvNeXt_Pruned(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], p=0.95, drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()
        pruned_dims = [dim if dim != 768 else int(dim * p) for dim in dims] # p here is scaling parameter, pruning ratio = 1-p
        pruned_dims = [32, 64, 128, 256]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, pruned_dims[0], kernel_size=4, stride=4),
            LayerNorm(pruned_dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(pruned_dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(pruned_dims[i], pruned_dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=pruned_dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(pruned_dims[-1], eps=1e-6)
        self.head = nn.Linear(pruned_dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


#%% perform pruning, load pruned weights to the pruned model


def generate_random_selected_indexes(state_dict, prune_factor=0.25):
    selected_indexes_dict = {}
    for key, param in state_dict.items():
        if param.shape[0] == 768:
            selected_indexes_dict[key] = random.sample(range(param.shape[0]), int(param.shape[0] * prune_factor))
    return selected_indexes_dict

def load_pruned_model_weights(original_model, pruned_model, selected_indexes_dict):
    original_state_dict = original_model.state_dict()
    pruned_state_dict = pruned_model.state_dict()

    for key in pruned_state_dict.keys():
        original_weight = original_state_dict[key]
        pruned_weight = pruned_state_dict[key]
        if original_weight.shape == pruned_weight.shape:
            pruned_state_dict[key] = original_weight.clone()
            print(key, ' loaded with original')
        elif original_weight.shape[0] == 768:
            selected_indexes = selected_indexes_dict[key]
            pruned_size_0 = pruned_weight.shape[0]
            pruned_size_1 = pruned_weight.shape[1] if len(original_weight.shape) > 1 else None
            if pruned_size_1:
                pruned_weight = original_weight[sorted(selected_indexes[-pruned_size_0:]), -pruned_size_1:]
            else:
                pruned_weight = original_weight[sorted(selected_indexes[-pruned_size_0:])]
            pruned_state_dict[key] = pruned_weight
            print(key, ' loaded with pruned')

    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model


def load_pruned_model_weights_keys_org(original_model, pruned_model, selected_indexes_dict,keys_to_pruned):
    original_state_dict = original_model.state_dict()
    pruned_state_dict = pruned_model.state_dict()
    keys_to_pruned = keys_to_pruned


    for key in pruned_state_dict.keys():
        original_weight = original_state_dict[key]
        pruned_weight = pruned_state_dict[key]
        if original_weight.shape == pruned_weight.shape:
            pruned_state_dict[key] = original_weight.clone()
            print(key, ' loaded with original')
        elif key in keys_to_pruned: #original_weight.shape[0] == 768:
            selected_indexes = selected_indexes_dict[key]
            pruned_size_0 = pruned_weight.shape[0]
            pruned_size_1 = pruned_weight.shape[1] if len(original_weight.shape) > 1 else None
            if pruned_size_1:
                if 'pwconv1' in key.split() :
                    key_prev = key.replace('pwconv1','dwconv')

                    sorted_indexes_prev = selected_indexes_dict[key_prev]
                elif 'pwconv2' in key.split() :
                    key_prev = key.replace('pwconv2','pwconv1')
                    sorted_indexes_prev = selected_indexes_dict[key_prev]                    
                pruned_weight = original_weight[sorted(selected_indexes[-pruned_size_0:]),sorted(sorted_indexes_prev[-pruned_size_1:])]
                print('key is {}, prev key is {}'.format(key, key_prev))

            else:
                pruned_weight = original_weight[sorted(selected_indexes[-pruned_size_0:])]
            pruned_state_dict[key] = pruned_weight
            print(key, ' loaded with pruned')

    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model


def load_pruned_model_weights_keys_chatgpt(original_model, pruned_model, selected_indexes_dict, keys_to_pruned):
    original_state_dict = original_model.state_dict()
    pruned_state_dict = pruned_model.state_dict()

    for key in pruned_state_dict.keys():
        original_weight = original_state_dict[key]
        pruned_weight = pruned_state_dict[key]

        if original_weight.shape == pruned_weight.shape:
            pruned_state_dict[key] = original_weight.clone()
            print(f"{key} loaded with original")
        elif key in keys_to_pruned:
            selected_indexes = selected_indexes_dict[key]
            pruned_size_0 = pruned_weight.shape[0]
            pruned_size_1 = pruned_weight.shape[1] if len(original_weight.shape) > 1 else None

            sorted_indexes_prev = None  # Initialize to prevent UnboundLocalError

            if pruned_size_1:
                if 'pwconv1' in key:
                    key_prev = key.replace('pwconv1', 'dwconv')
                elif 'pwconv2' in key:
                    key_prev = key.replace('pwconv2', 'pwconv1')
                else:
                    key_prev = None  # If key does not match, set None

                if key_prev and key_prev in selected_indexes_dict:  # Ensure key exists before access
                    sorted_indexes_prev = selected_indexes_dict[key_prev]
                    pruned_weight = original_weight[
                        sorted(selected_indexes[-pruned_size_0:]),
                        sorted(sorted_indexes_prev[-pruned_size_1:])
                    ]
                    print(f"Key: {key}, Previous Key: {key_prev}")
                else:
                    print(f"Warning: {key_prev} not found in selected_indexes_dict, skipping index pruning.")

            else:
                pruned_weight = original_weight[sorted(selected_indexes[-pruned_size_0:])]

            pruned_state_dict[key] = pruned_weight
            print(f"{key} loaded with pruned")

    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model


def load_pruned_model_weights_keys(original_model, pruned_model, selected_indexes_dict, keys_to_pruned):
    original_state_dict = original_model.state_dict()
    pruned_state_dict = pruned_model.state_dict()

    for key in pruned_state_dict.keys():
        original_weight = original_state_dict[key]
        pruned_weight = pruned_state_dict[key]

        if original_weight.shape == pruned_weight.shape:
            pruned_state_dict[key] = original_weight.clone()
            print(f"{key} loaded with original")
        elif key in keys_to_pruned:
            selected_indexes = selected_indexes_dict[key]
            pruned_size_0 = pruned_weight.shape[0]
            pruned_size_1 = pruned_weight.shape[1] if len(original_weight.shape) > 1 else None

            sorted_indexes_prev = None  # Initialize to prevent UnboundLocalError

            if pruned_size_1:
                if 'pwconv1' in key:
                    key_prev = key.replace('pwconv1', 'dwconv')
                elif 'pwconv2' in key:
                    key_prev = key.replace('pwconv2', 'pwconv1')
                else:
                    key_prev = None  # If key does not match, set None

                if key_prev and key_prev in selected_indexes_dict:
                    sorted_indexes_prev = selected_indexes_dict[key_prev]

                    # **Debugging Prints**
                    print(f"\nKey: {key}, Previous Key: {key_prev}")
                    print(f"Original Shape: {original_weight.shape}, Pruned Shape: {pruned_weight.shape}")
                    print(f"Selected Indexes: {len(selected_indexes)}, Sorted Indexes Prev: {len(sorted_indexes_prev)}")
                    print(f"Pruned Size 0: {pruned_size_0}, Pruned Size 1: {pruned_size_1}")

                    # **Shape Check Before Indexing**
                    assert len(selected_indexes) >= pruned_size_0, f"Mismatch in selected_indexes for {key}"
                    assert len(sorted_indexes_prev) >= pruned_size_1, f"Mismatch in sorted_indexes_prev for {key_prev}"

                    pruned_weight = original_weight[
                        sorted(selected_indexes[-pruned_size_0:]), :][:,
                        sorted(sorted_indexes_prev[-pruned_size_1:])
                    ]
                    print(f"Key: {key} successfully pruned using {key_prev}")

                else:
                    print(f"Warning: {key_prev} not found in selected_indexes_dict, skipping index pruning.")

            else:
                assert len(selected_indexes) >= pruned_size_0, f"Mismatch in selected_indexes for {key}"
                pruned_weight = original_weight[sorted(selected_indexes[-pruned_size_0:])]

            pruned_state_dict[key] = pruned_weight
            print(f"{key} loaded with pruned")

    pruned_model.load_state_dict(pruned_state_dict)
    return pruned_model


#%% geenrate pruned model from originally pre-trained model
 
model_original = convnext_tiny()
state_dict_path = '/home/arshdeep/ConvNeXt/checkpoints/convnext_tiny_1k_224_ema.pth' #original weights
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu')) 

model_original.load_state_dict(state_dict['model'])

arch_pruned = ConvNeXt_Pruned( num_classes =10)
# Generate the computation graph
# summary(model_original, input_size = (1,3,224,224))
# index = generate_random_selected_indexes(model_original.state_dict())
# index = generate_random_selected_indexes(model_original.state_dict())  # randomly generated indexex
with open('/home/arshdeep/ConvNeXt/sorted_index_imagenet_alllayers/OP/sorted_index_convnext_op.pkl', 'rb') as f:
    index = pickle.load(f)
    

keys_to_pruned = list(state_dict['model'].keys())


'''
['downsample_layers.3.1.weight', 'downsample_layers.3.1.bias','stages.3.0.gamma',
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
model_pruned = load_pruned_model_weights_keys(model_original, arch_pruned, index, keys_to_pruned)  # pruned model

#%% save model dict
'''
import os
save_dir = '/home/arshdeep/ConvNeXt/checkpoints/new_with_pwconv/'
#os.makedirs(save_dir, exist_ok=True)
torch.save(model_pruned.state_dict(), os.path.join(save_dir, 'pruned_all32_64_128_256_op_convnext_tiny_1k_224_ema.pth'))
'''
#%%
print(summary(model_original, input_size = (1,3,224,224)))
print(summary(arch_pruned, input_size = (1,3,224,224)))
w_org = model_original.state_dict()
w_pruned = model_pruned.state_dict()

#%% pruning with selected keys
'''
# list_all_pruned_keys = ['downsample_layers.3.1.weight', 'downsample_layers.3.1.bias', 'stages.3.0.gamma', 'stages.3.0.dwconv.weight', 'stages.3.0.dwconv.bias', 'stages.3.0.norm.weight', 'stages.3.0.norm.bias', 'stages.3.0.pwconv2.weight', 'stages.3.0.pwconv2.bias', 'stages.3.1.gamma', 'stages.3.1.dwconv.weight', 'stages.3.1.dwconv.bias', 'stages.3.1.norm.weight', 'stages.3.1.norm.bias', 'stages.3.1.pwconv2.weight', 'stages.3.1.pwconv2.bias', 'stages.3.2.gamma', 'stages.3.2.dwconv.weight', 'stages.3.2.dwconv.bias', 'stages.3.2.norm.weight', 'stages.3.2.norm.bias', 'stages.3.2.pwconv2.weight', 'stages.3.2.pwconv2.bias', 'norm.weight', 'norm.bias']

list_pruned_layers = ['downsample_layers.3.1.weight','stages.3.0.dwconv.weight', 'stages.3.0.pwconv1.weight','stages.3.0.pwconv2.weight','stages.3.1.dwconv.weight', 'stages.3.1.pwconv1.weight','stages.3.1.pwconv2.weight','stages.3.2.dwconv.weight', 'stages.3.2.pwconv1.weight','stages.3.2.pwconv2.weight', '']
# tensor_reshaped  = Q.view(3072, 768, 1, 1) #Q.unsqueeze(2).unsqueeze(3)
weights = w_org

def channel_wise_score(key, w_org):
    weights = w_org
    z = key.split('.')
    W_2D = weights[key]
    if ('pwconv1' in z) or ('pwconv2' in z):
        W_2D = W_2D.unsqueeze(2).unsqueeze(3)
                
    W_2D = W_2D.numpy()
    W = np.reshape(W_2D,(np.shape(W_2D)[0],np.shape(W_2D)[1],np.shape(W_2D)[2]*np.shape(W_2D)[3]))
    print(W.shape)  # W Shape: [FILTERS, CHANNELS, HEIGHT, WIDTH]
    score = operator_norm_pruning(W)   
    return np.array(score)

High_key = ['stages.3.0.dwconv.weight','stages.3.0.pwconv1.weight', 'stages.3.0.pwconv2.weight','stages.3.1.dwconv.weight','stages.3.1.pwconv1.weight', 'stages.3.1.pwconv2.weight','stages.3.2.dwconv.weight','stages.3.2.pwconv1.weight', 'stages.3.2.pwconv2.weight','downsample_layers.3.1.weight']



# High_key = ['downsample_layers.3.1.weight']

sorted_index_dict  = {}

for key in High_key:    
    score = channel_wise_score(key, w_org)
    sorted_index = np.argsort(score) # low to high score, high is more important
    sorted_index_dict[key] = sorted_index

#%%




key_dwconv_stage3_0 = ['stages.3.0.dwconv.bias','stages.3.0.dwconv.gamma','stages.3.0.norm.bias','stages.3.0.norm.weight']

key_dwconv_stage3_1 = ['stages.3.1.dwconv.bias','stages.3.1.dwconv.gamma','stages.3.1.norm.bias','stages.3.1.norm.weight']

key_dwconv_stage3_2 = ['stages.3.2.dwconv.bias','stages.3.2.dwconv.gamma','stages.3.2.norm.bias','stages.3.2.norm.weight']






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
#%%
import os
os.chdir('/home/arshdeep/ConvNeXt/sorted_index/OP/')


import pickle

# Sample dictionary
# data = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# Save dictionary to a file
with open('sorted_index_convnext.pkl', 'wb') as f:
    pickle.dump(sorted_index_dict, f)

'''

#%%

import torch
import timm
from thop import profile

# Load ConvNeXt-Tiny model
# model = timm.create_model("convnext_tiny", pretrained=False)

# Dummy input of shape (1,3,224,224)
input_tensor = torch.randn(1, 3, 224, 224)

# Compute FLOPs and parameters
macs, params = profile(model_original, inputs=(input_tensor,))
print(f"MACs: {macs / 1e9:.2f} GMACs")  # Convert to GMACs
print(f"Params: {params / 1e6:.2f} M")   # Convert to Million parameters
