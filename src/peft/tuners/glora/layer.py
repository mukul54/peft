# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(56)  # For reproducibility

class GLoraLayer:
    """
    Base implementation of GLoRA (Group LoRA) support tensors.
    This class defines the core functionality used by the specialized layers.
    """
    def __init__(self, in_features: int, out_features: int, r: int, adapter_name: str, **kwargs):
        self.r = {}
        self.r[adapter_name] = r
        self.glora_Ad, self.glora_Au = self.make_param((out_features, in_features), f'LoRA_{r}')
        self.glora_Bd, self.glora_Bu = self.make_param((out_features, in_features), f'LoRA_{r}')
        self.glora_Cd, self.glora_Cu = self.make_param((in_features, 1), f'LoRA_{r}')
        self.glora_D = nn.Parameter(torch.zeros(out_features))
        self.glora_E = nn.Parameter(torch.zeros(out_features))
        self.eval_config = None
        nn.init.kaiming_uniform_(self.glora_Au, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Bu, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.glora_Cu, a=math.sqrt(5))
        # Mark the weight as unmerged
        self.merged = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        config_A_B = [f'LoRA_{r}', 'vector', 'constant', 'none']
        config_C = [f'LoRA_{r}', 'vector', 'none']
        config_D_E = ['constant', 'none', 'vector']
        self.configs = []
        for A in config_A_B:
            for B in config_A_B:
                for C in config_C:
                    for D in config_D_E:
                        for E in config_D_E:
                            config = {'A':A,'B':B,'C':C,'D':D,'E':E}
                            self.configs.append(config)
    
    def make_param(self, shape, config=None):
        """
        Create low-rank parameter matrices based on configuration.
        
        Args:
            shape: Tuple of (output_dim, input_dim)
            config: Configuration string describing the parameter type
            
        Returns:
            Parameter tensor(s) based on the specified configuration
        """
        if 'LoRA' in config:
            out_feature = shape[0]
            in_feature = shape[1]
            try:
                rank = int(config.split('_')[1])
            except:
                rank = 4
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape))
    
    def prepare_path(self, config, Xd, Xu=None):
        """
        Prepare the tensors for the specified path configuration.
        
        Args:
            config: Configuration string for this tensor
            Xd: First tensor (or only tensor for single-tensor configs)
            Xu: Second tensor for two-tensor configurations (optional)
            
        Returns:
            Tensor prepared according to the configuration
        """
        device = self.weight.device  # Get the device directly from weight
        
        # Ensure input tensors are on the correct device first
        Xd = Xd.to(device)
        if Xu is not None:
            Xu = Xu.to(device)
            
            if 'LoRA' in config:
                rank = int(config.split('_')[1])
                X = torch.matmul(Xd[:,:rank], Xu[:rank, :])
            elif 'vector' in config:
                X = Xd[:,0].unsqueeze(1)
            elif 'constant' in config:
                X = Xd[0,0]
            elif 'none' in config:
                X = torch.zeros(Xd.shape[0], Xu.shape[1], device=device)  # Create directly on device
            else:
                raise ValueError(f"Unknown config: {config}")
        else:
            if 'vector' in config:
                X = Xd
            elif 'constant' in config:
                X = Xd[0]
            elif 'none' in config:
                X = torch.zeros(1, device=device)  # Create directly on device
            else:
                raise ValueError(f"Unknown config: {config}")
                
        # No need for extra .to(device) since everything should already be on the right device
        return X


class Linear(nn.Linear, GLoraLayer):
    """GLora implementation for nn.Linear layers"""
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        GLoraLayer.__init__(self, in_features=in_features, out_features=out_features, r=r, adapter_name=adapter_name)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        nn.Linear.reset_parameters(self)
        self.active_adapter = adapter_name
        # Don't call .to() here as it can be expensive and redundant
        # The device will be set when the module is moved by the parent

    def merge(self):
        """
        Merge the GLoRA parameters into the base weights.
        After merging, the layer can be used as a standard nn.Linear layer.
        """
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
            
        # If eval_config is not set, use the first config as default
        if self.eval_config is None and len(self.configs) > 0:
            self.eval_config = self.configs[0]
            
        path_config = self.eval_config
        A = self.prepare_path(path_config['A'], self.glora_Ad, self.glora_Au).to(self.weight.dtype)
        B = self.prepare_path(path_config['B'], self.glora_Bd, self.glora_Bu).to(self.weight.dtype)
        C = self.prepare_path(path_config['C'], self.glora_Cd, self.glora_Cu).to(self.weight.dtype)
        D = self.prepare_path(path_config['D'], self.glora_D).to(self.weight.dtype)
        E = self.prepare_path(path_config['E'], self.glora_E).to(self.weight.dtype)
        
        # Merge into weight
        self.weight.data += self.weight*A + B
        
        # Merge into bias
        if torch.is_tensor(self.bias):
            self.bias.data += self.bias*D + E+torch.matmul(self.weight, C).squeeze()
        else:
            self.bias = nn.Parameter(E+torch.matmul(self.weight, C).squeeze())
            
        self.merged = True

    def forward(self, x: torch.Tensor):
        """
        Forward pass with GLoRA adaptation
        """
        previous_dtype = x.dtype
        device = self.weight.device
        dtype = self.weight.dtype
        
        # Move input to the same device as weights
        x = x.to(device)
        
        # Select path configuration
        if self.eval_config is not None:
            path_config = self.eval_config
        else:
            path_config = random.choice(self.configs)
            
        # Prepare all tensors and ensure they're on the correct device and dtype
        A = self.prepare_path(path_config['A'], self.glora_Ad, self.glora_Au).to(device=device, dtype=dtype)
        B = self.prepare_path(path_config['B'], self.glora_Bd, self.glora_Bu).to(device=device, dtype=dtype)
        C = self.prepare_path(path_config['C'], self.glora_Cd, self.glora_Cu).to(device=device, dtype=dtype)
        D = self.prepare_path(path_config['D'], self.glora_D).to(device=device, dtype=dtype)
        E = self.prepare_path(path_config['E'], self.glora_E).to(device=device, dtype=dtype)
        
        # Compute with all tensors guaranteed to be on same device and dtype
        weight_sum = self.weight + self.weight*A + B
        
        if torch.is_tensor(self.bias):
            # Make sure bias is on the right device
            bias = self.bias.to(device=device, dtype=dtype)
            bias_sum = bias + bias*D + E + torch.matmul(self.weight, C).squeeze()
            result = F.linear(x, weight_sum, bias=bias_sum)
        else:
            bias_sum = E + torch.matmul(self.weight, C).squeeze()
            result = F.linear(x, weight_sum, bias=bias_sum)
        
        # Restore the original dtype
        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)
            
        return result
    
    def enable_adapters(self):
        """Enable GLoRA adapters in this layer"""
        self.merged = False
        
    def disable_adapters(self):
        """Disable GLoRA adapters in this layer (will merge first if not merged)"""
        if not self.merged:
            self.merge()
