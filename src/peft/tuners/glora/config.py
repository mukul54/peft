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

from dataclasses import dataclass, field
from typing import Optional, Union, List

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class GLoraConfig(PeftConfig):
    """
    This is the configuration class for GLoRA (Group Low-Rank Adaptation) method.

    Args:
        r (`int`): GLoRA rank dimension.
        target_modules (`Union[List[str], str]`): The names of the modules to apply GLoRA to.
        inference_mode (`bool`): If True, model weights are not trainable. Default: False.
        task_type (`str`): Task for which this configuration is used. Default: None.
    """

    r: int = field(default=4, metadata={"help": "GLoRA rank dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with GLoRA."
            "For example, ['q_proj', 'v_proj'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})
    task_type: Optional[str] = field(default=None, metadata={"help": "Task type"})

    def __post_init__(self):
        self.peft_type = PeftType.GLORA
