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

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
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


class GLoraModel(torch.nn.Module):
    """
    Creates Generalized Low Rank Adapter (GLora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`GLoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, GLoraConfig
        >>> from peft import GLoraModel, GLoraConfig

        >>> config = GLoraConfig(
        ...     peft_type="GLORA",
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

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_glora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_glora_as_trainable(self.model)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _check_target_module_exists(self, glora_config, key):
        if isinstance(glora_config.target_modules, str):
            target_module_found = re.fullmatch(glora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in glora_config.target_modules)
        return target_module_found

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r
        }
        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
        new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def _find_and_replace(self, adapter_name):
        glora_config = self.peft_config[adapter_name]
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(glora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            new_module = self._create_new_module(glora_config, adapter_name, target)
            self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {glora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "glora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    @staticmethod
    def _prepare_glora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            if child.bias is not None:
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "glora_" in name:
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

    def merge_and_unload(self, progressbar: bool = False):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (bool): whether to show a progressbar indicating the unload and merge process

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-glora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(progressbar=progressbar)

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False):

        key_list = [key for key, _ in self.model.named_modules() if "glora" not in key]
        for key in tqdm(key_list, disable=not progressbar ):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, GLoraLayer):
                bias = True
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

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
        if self.eval_config is not None:
            path_config = self.eval_config
        else:
            path_config = random.choice(self.configs)
        A = self.prepare_path(path_config['A'], self.glora_Ad, self.glora_Au).to(self.weight.dtype)
        B = self.prepare_path(path_config['B'], self.glora_Bd, self.glora_Bu).to(self.weight.dtype)
        C = self.prepare_path(path_config['C'], self.glora_Cd, self.glora_Cu).to(self.weight.dtype)
        D = self.prepare_path(path_config['D'], self.glora_D).to(self.weight.dtype)
        E = self.prepare_path(path_config['E'], self.glora_E).to(self.weight.dtype)
        if torch.is_tensor(self.bias):
            result = F.linear(x, self.weight + self.weight*A + B, bias=self.bias + self.bias*D + E+torch.matmul(self.weight, C).squeeze())
        else:
            result = F.linear(x, self.weight + self.weight*A + B, bias=E+torch.matmul(self.weight, C).squeeze())
        result = result.to(previous_dtype)

        return result
    
    def prepare_path(self, config, Xd, Xu=None):
        if Xu is not None:
            if 'LoRA' in config:
                rank = int(config.split('_')[1])
                X = torch.matmul(Xd[:,:rank], Xu[:rank, :])
            elif 'vector' in config:
                X = Xd[:,0].unsqueeze(1)
            elif 'constant' in config:
                X = Xd[0,0]
            elif 'none' in config:
                X = torch.zeros(Xd.shape[0], Xu.shape[1]).to(self.weight.device)
            else:
                raise ValueError
        else:
            if 'vector' in config:
                X = Xd
            elif 'constant' in config:
                X = Xd[0]
            elif 'none' in config:
                X = torch.zeros(1).to(self.weight.device)
            else:
                raise ValueError
        return X