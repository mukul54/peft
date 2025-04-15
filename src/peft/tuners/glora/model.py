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

import re
import warnings
import torch
from torch import nn
from typing import List, Optional, Union
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTuner
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
)

from .config import GLoraConfig
from .layer import Linear, GLoraLayer


def mark_only_glora_as_trainable(model: nn.Module) -> None:
    """
    Mark only the GLoRA parameters as trainable, freezing all other parameters.
    
    Args:
        model: The model with GLoRA layers
    """
    for n, p in model.named_parameters():
        if "glora_" not in n:
            p.requires_grad = False


class GLoraModel(BaseTuner):
    """
    Creates a GLoRA (Group Low-Rank Adaptation) model from a pretrained transformers model.

    Args:
        model (`torch.nn.Module`): The model to be adapted.
        config (`GLoraConfig`): The configuration of the GLoRA model.
        adapter_name (`str`): The name of the adapter, defaults to "default".

    Example:
    
    ```py
    >>> from transformers import AutoModelForCausalLM
    >>> from peft import GLoraConfig, GLoraModel
    >>> 
    >>> config = GLoraConfig(
    ...     task_type="CAUSAL_LM",
    ...     r=8,
    ...     target_modules=["q_proj", "v_proj"],
    ... )
    >>> 
    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-hf")
    >>> glora_model = GLoraModel(model, config)
    ```
    """

    def __init__(self, model, peft_config, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)

    def _prepare_adapter_config(self, peft_config, model_config):
        """
        Prepares the adapter config.
        
        Args:
            peft_config: The PEFT configuration
            model_config: The model configuration
            
        Returns:
            The prepared adapter configuration
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
        Check if the module key matches any target module pattern.
        
        Args:
            peft_config: The PEFT configuration
            key: The module key to check
            
        Returns:
            True if the key matches a target module, False otherwise
        """
        if isinstance(peft_config.target_modules, str):
            target_modules = [peft_config.target_modules]
        else:
            target_modules = peft_config.target_modules

        return any(key.endswith(target_key) for target_key in target_modules)

    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent, current_key):
        """
        Creates and replaces a target module with a GLoRA layer.
        
        Args:
            peft_config: The PEFT configuration
            adapter_name: The name of the adapter
            target: The target module to replace
            target_name: The name of the target module
            parent: The parent module
            current_key: The full key of the current module
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
        
        Args:
            model: The model with GLoRA layers
        """
        for param in model.parameters():
            param.requires_grad = False

        # Set requires_grad for GLoRA layers
        for name, param in model.named_parameters():
            if "glora_" in name:
                param.requires_grad = True

    def enable_adapter_layers(self):
        """
        Enables adapter layers.
        """
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.enable_adapters()

    def disable_adapter_layers(self):
        """
        Disables adapter layers.
        """
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.disable_adapters()
    
    def set_adapter(self, adapter_names):
        """
        Set the active adapters.
        
        Args:
            adapter_names (str or List[str]): Name of the adapter(s) to be activated.
            
        Returns:
            The active adapter names
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
        
        Args:
            inference: Whether to set inference mode in the config
            
        Returns:
            Dictionary containing the configuration
        """
        config_dict = {}
        for adapter_name, adapter_config in self.peft_config.items():
            config_dict[adapter_name] = adapter_config.to_dict()
            if inference:
                config_dict[adapter_name]["inference_mode"] = True
        return config_dict

    def merge_and_unload(self, progressbar: bool = False):
        """
        Merge the GLoRA layers into the base model weights and unload the GLoRA adapter.
        
        This is useful for deploying the model after training is complete.
        
        Args:
            progressbar: Whether to show a progress bar during merging
            
        Returns:
            The base model with GLoRA weights merged in
        """
        key_list = [key for key, _ in self.model.named_modules() if "glora" not in key]
        for key in tqdm(key_list, disable=not progressbar, desc="Merging GLoRA adapters"):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
                
            if isinstance(target, GLoraLayer) or isinstance(target, Linear):
                bias = hasattr(target, "bias") and target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                
                # Set eval_config if not already set
                if target.eval_config is None and hasattr(target, "configs") and len(target.configs) > 0:
                    target.eval_config = target.configs[0]
                
                # Merge the weights
                target.merge()
                
                # Copy the merged weights
                new_module.weight.data = target.weight.data.clone()
                if bias:
                    new_module.bias.data = target.bias.data.clone()
                
                # Replace with the new module
                setattr(parent, target_name, new_module)
                
            # Handle any modules_to_save
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])
        
        return self.model
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Prepare inputs for generation. This method is required for compatibility with
        PeftModelForCausalLM when using get_peft_model().
        
        This method passes through to the underlying model's prepare_inputs_for_generation
        method if it exists, otherwise it returns the kwargs unchanged.
        
        Args:
            *args: Arguments to pass to the base model
            **kwargs: Keyword arguments to pass to the base model
            
        Returns:
            The prepared inputs
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
