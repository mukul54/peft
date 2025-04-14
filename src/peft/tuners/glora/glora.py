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
import re
import warnings
import random
random.seed(56)
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner

from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose
)


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class GLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`GLoraModel`].

    Args:
        r (`int`): GLora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=4, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.GLORA


class GLoraModel(BaseTuner):
    """
    Creates a GLoRA (Group Lora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`GLoraConfig`]): The configuration of the GLoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import GLoraConfig, GLoraModel

        >>> config = GLoraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     target_modules=["q", "v"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> glora_model = GLoraModel(config, model)
        ```

        ```py
        >>> import transformers
        >>> from peft import GLoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = GLoraConfig(
        ...     r=4, target_modules=target_modules, task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> glora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`GLoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, peft_config, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)

    def _prepare_adapter_config(self, peft_config, model_config):
        """
        Prepares the adapter config.
        """
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError(
                    f"Target modules are not specified in the config and no default target modules exist for "
                    f"{model_config['model_type']}."
                )
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def _check_target_module_exists(self, peft_config, key):
        """
        A helper method to check if the passed module's key name matches any of the target modules in the
        peft_config.target_modules list. If it does, return True, else return False.
        
        Args:
            peft_config (PeftConfig): The adapter config.
            key (str): The module's key name.
        """
        if isinstance(peft_config.target_modules, str):
            target_modules = [peft_config.target_modules]
        else:
            target_modules = peft_config.target_modules

        return any(key.endswith(target_key) for target_key in target_modules)

    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent, current_key):
        """
        Creates and replaces a target module with a GLoRA layer.
        """
        # Create the replacement GLoRA module
        bias = hasattr(target, "bias") and target.bias is not None
        
        # Only use attributes that exist in GLoraConfig
        kwargs = {
            "r": peft_config.r,
        }
        
        # Create the GLoRA module based on the target type
        if isinstance(target, nn.Linear):
            # Use the correct parameters for the Linear constructor
            glora_layer = Linear(
                adapter_name=adapter_name, 
                in_features=target.in_features, 
                out_features=target.out_features, 
                **kwargs
            )
            
            # Copy the weights from the original module
            glora_layer.weight.data = target.weight.data.clone()
            if bias:
                glora_layer.bias.data = target.bias.data.clone()
                
            # Update the list of modules we've modified
            self.targeted_module_names.append(current_key)
            
            # Replace the original module with the GLoRA module
            parent._modules[target_name] = glora_layer
        else:
            # For other layer types, customize as needed
            return

    def _mark_only_adapters_as_trainable(self, model):
        """
        Marks only the GLoRA adapters as trainable.
        """
        for param in model.parameters():
            param.requires_grad = False

        # Set requires_grad for GLoRA layers
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def enable_adapter_layers(self):
        """
        Enables adapter layers
        """
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.enable_adapters()

    def disable_adapter_layers(self):
        """
        Disables adapter layers
        """
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.disable_adapters()
    
    def set_adapter(self, adapter_names):
        """
        Set the active adapters.
        
        Args:
            adapter_names (str or List[str]): Name of the adapter(s) to be activated.
        """
        # Store the active adapter(s)
        self.active_adapter = adapter_names
        
        # Enable the active adapters in all Linear layers
        for module in self.model.modules():
            if isinstance(module, Linear):
                if isinstance(adapter_names, list) and len(adapter_names) > 0:
                    module.active_adapter = adapter_names[0]
                else:
                    module.active_adapter = adapter_names
                
        return adapter_names
        
    def get_peft_config_as_dict(self, inference: bool = False):
        """
        Returns the config as a dictionary for saving.
        """
        config_dict = {}
        for adapter_name, adapter_config in self.peft_config.items():
            config_dict[adapter_name] = adapter_config.to_dict()
        return config_dict

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Prepare inputs for generation. This method is required for compatibility with
        PeftModelForCausalLM when using get_peft_model().
        
        This method passes through to the underlying model's prepare_inputs_for_generation
        method if it exists, otherwise it returns the kwargs unchanged.
        """
        if hasattr(self.model, "prepare_inputs_for_generation"):
            return self.model.prepare_inputs_for_generation(*args, **kwargs)
        # If no prepare_inputs_for_generation exists, create a default implementation
        # that matches the standard transformer behavior
        input_dict = kwargs
        if args and not kwargs:
            input_dict = dict(zip(['input_ids', 'attention_mask', 'past_key_values'], args))
        return input_dict
    
    @property
    def device(self):
        """
        Returns the device on which the model is located.
        This is needed for compatibility with scripts that rely on model.device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return next(self.model.parameters()).device

    @property
    def generation_config(self):
        """
        Returns the generation configuration for text generation.
        This is needed for compatibility with PeftModelForCausalLM.generate().
        """
        if hasattr(self.model, "generation_config"):
            return self.model.generation_config
        # If the base model doesn't have generation_config, create a default one
        from transformers import GenerationConfig
        return GenerationConfig()
        
    @generation_config.setter
    def generation_config(self, value):
        """
        Sets the generation configuration.
        This is needed for compatibility with PeftModelForCausalLM.generate().
        """
        if hasattr(self.model, "generation_config"):
            self.model.generation_config = value
            
    def generate(self, *args, **kwargs):
        """
        Pass through to model.generate().
        """
        return self.model.generate(*args, **kwargs)
        
    @property
    def config(self):
        """
        Returns the configuration object of the base model.
        This is needed for compatibility with the PeftModel forward method.
        """
        if hasattr(self.model, "config"):
            return self.model.config
        return None

# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_glora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "glora_" not in n:
            p.requires_grad = False

class GLoraLayer:
    def __init__(self, in_features: int, out_features: int, r: int, adapter_name: str,  **kwargs):
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
        if 'LoRA' in config:
            out_feature = shape[0]
            in_feature = shape[1]
            try:
                rank = int(config.split('_')[1])
            except:
                rank = 4
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape))


class Linear(nn.Linear, GLoraLayer):
    # GLora implemented in a dense layer
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
        self.to(self.weight.device)

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        path_config = self.eval_config
        A = self.prepare_path(path_config['A'], self.glora_Ad, self.glora_Au).to(self.weight.dtype)
        B = self.prepare_path(path_config['B'], self.glora_Bd, self.glora_Bu).to(self.weight.dtype)
        C = self.prepare_path(path_config['C'], self.glora_Cd, self.glora_Cu).to(self.weight.dtype)
        D = self.prepare_path(path_config['D'], self.glora_D).to(self.weight.dtype)
        E = self.prepare_path(path_config['E'], self.glora_E).to(self.weight.dtype)
        self.weight.data += self.weight*A + B
        if torch.is_tensor(self.bias):
            self.bias.data += self.bias*D + E+torch.matmul(self.weight, C).squeeze()
        else:
            self.bias = nn.Parameter(E+torch.matmul(self.weight, C).squeeze())
        self.merged = True

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        # Move input to the same device as weights
        x = x.to(self.weight.device)
        if self.eval_config is not None:
            path_config = self.eval_config
        else:
            path_config = random.choice(self.configs)
        A = self.prepare_path(path_config['A'], self.glora_Ad, self.glora_Au).to(self.weight.dtype)
        B = self.prepare_path(path_config['B'], self.glora_Bd, self.glora_Bu).to(self.weight.dtype)
        C = self.prepare_path(path_config['C'], self.glora_Cd, self.glora_Cu).to(self.weight.dtype)
        D = self.prepare_path(path_config['D'], self.glora_D).to(self.weight.dtype)
        E = self.prepare_path(path_config['E'], self.glora_E).to(self.weight.dtype)
        # Ensure all tensors are on the same device and dtype before any operation
        device = self.weight.device
        dtype = self.weight.dtype
        
        # Move all individual tensors to the same device first
        A = A.to(device=device, dtype=dtype)
        B = B.to(device=device, dtype=dtype)
        C = C.to(device=device, dtype=dtype)
        D = D.to(device=device, dtype=dtype)
        E = E.to(device=device, dtype=dtype)
        
        # Now compute with all tensors guaranteed to be on same device
        weight_sum = self.weight + self.weight*A + B
        
        if torch.is_tensor(self.bias):
            # Properly handle the bias parameter without reassigning it
            bias_to_use = self.bias.to(device=device, dtype=dtype)  # Create temporary tensor on right device
            bias_sum = bias_to_use + bias_to_use*D + E + torch.matmul(self.weight, C).squeeze()
            result = F.linear(x, weight_sum, bias=bias_sum)
        else:
            bias_sum = E + torch.matmul(self.weight, C).squeeze()
            result = F.linear(x, weight_sum, bias=bias_sum)
        result = result.to(previous_dtype)

        return result
    
    def prepare_path(self, config, Xd, Xu=None):
        device = self.weight.device  # Get the device of the weight
        
        if Xu is not None:
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
                raise ValueError
        else:
            if 'vector' in config:
                X = Xd
            elif 'constant' in config:
                X = Xd[0]
            elif 'none' in config:
                X = torch.zeros(1, device=device)  # Create directly on device
            else:
                raise ValueError
                
        return X.to(device)  # Ensure return tensor is on the correct device