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

__version__ = "0.15.2.dev0"

from .auto import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForFeatureExtraction,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForTokenClassification,
)
from .config import PeftConfig, PromptLearningConfig
from .mapping import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PEFT_TYPE_TO_MIXED_MODEL_MAPPING,
    PEFT_TYPE_TO_TUNER_MAPPING,
    get_peft_config,
    inject_adapter_in_model,
)
from .mapping_func import get_peft_model
from .mixed_model import PeftMixedModel
from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
    get_layer_status,
    get_model_status,
)
from .tuners import (
    AdaLoraConfig,
    AdaLoraModel,
    AdaptionPromptConfig,
    AdaptionPromptModel,
    BOFTConfig,
    BOFTModel,
    BoneConfig,
    BoneModel,
    CPTConfig,
    CPTEmbedding,
    EvaConfig,
    FourierFTConfig,
    FourierFTModel,
    HRAConfig,
    HRAModel,
    IA3Config,
    IA3Model,
    LNTuningConfig,
    LNTuningModel,
    LoftQConfig,
    LoHaConfig,
    LoHaModel,
    LoKrConfig,
    LoKrModel,
    LoraConfig,
    LoraModel,
    LoraRuntimeConfig,
    GLoraModel,
    GLoraConfig,
    MultitaskPromptTuningConfig,
    MultitaskPromptTuningInit,
    OFTConfig,
    OFTModel,
    PolyConfig,
    PolyModel,
    PrefixEncoder,
    PrefixTuningConfig,
    PromptEmbedding,
    PromptEncoder,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptTuningConfig,
    PromptTuningInit,
    TrainableTokensConfig,
    TrainableTokensModel,
    VBLoRAConfig,
    VBLoRAModel,
    VeraConfig,
    VeraModel,
    XLoraConfig,
    XLoraModel,
    get_eva_state_dict,
    initialize_lora_eva_weights,
)
from .utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    PeftType,
    TaskType,
    bloom_model_postprocess_past_key_value,
    cast_mixed_precision_params,
    get_peft_model_state_dict,
    load_peft_weights,
    prepare_model_for_kbit_training,
    replace_lora_weights_loftq,
    set_peft_model_state_dict,
    shift_tokens_right,
)


__all__ = [
    "MODEL_TYPE_TO_PEFT_MODEL_MAPPING",
    "PEFT_TYPE_TO_CONFIG_MAPPING",
    "PEFT_TYPE_TO_MIXED_MODEL_MAPPING",
    "PEFT_TYPE_TO_TUNER_MAPPING",
    "TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING",
    "AdaLoraConfig",
    "AdaLoraModel",
    "AdaptionPromptConfig",
    "AdaptionPromptModel",
    "AutoPeftModel",
    "AutoPeftModelForCausalLM",
    "AutoPeftModelForFeatureExtraction",
    "AutoPeftModelForQuestionAnswering",
    "AutoPeftModelForSeq2SeqLM",
    "AutoPeftModelForSequenceClassification",
    "AutoPeftModelForTokenClassification",
    "BOFTConfig",
    "BOFTModel",
    "BoneConfig",
    "BoneModel",
    "CPTConfig",
    "CPTEmbedding",
    "EvaConfig",
    "FourierFTConfig",
    "FourierFTModel",
    "HRAConfig",
    "HRAModel",
    "IA3Config",
    "IA3Model",
    "LNTuningConfig",
    "LNTuningModel",
    "LoHaConfig",
    "LoHaModel",
    "LoKrConfig",
    "LoKrModel",
    "LoftQConfig",
    "LoraConfig",
    "LoraModel",
    "LoraRuntimeConfig",
    "GLoraModel",
    "GLoraConfig",
    "MultitaskPromptTuningConfig",
    "MultitaskPromptTuningInit",
    "OFTConfig",
    "OFTModel",
    "PeftConfig",
    "PeftMixedModel",
    "PeftModel",
    "PeftModelForCausalLM",
    "PeftModelForFeatureExtraction",
    "PeftModelForQuestionAnswering",
    "PeftModelForSeq2SeqLM",
    "PeftModelForSequenceClassification",
    "PeftModelForTokenClassification",
    "PeftType",
    "PolyConfig",
    "PolyModel",
    "PrefixEncoder",
    "PrefixTuningConfig",
    "PromptEmbedding",
    "PromptEncoder",
    "PromptEncoderConfig",
    "PromptEncoderReparameterizationType",
    "PromptLearningConfig",
    "PromptTuningConfig",
    "PromptTuningInit",
    "TaskType",
    "TrainableTokensConfig",
    "TrainableTokensModel",
    "VBLoRAConfig",
    "VBLoRAConfig",
    "VBLoRAModel",
    "VeraConfig",
    "VeraModel",
    "XLoraConfig",
    "XLoraModel",
    "bloom_model_postprocess_past_key_value",
    "cast_mixed_precision_params",
    "get_eva_state_dict",
    "get_layer_status",
    "get_model_status",
    "get_peft_config",
    "get_peft_model",
    "get_peft_model_state_dict",
    "initialize_lora_eva_weights",
    "inject_adapter_in_model",
    "load_peft_weights",
    "prepare_model_for_kbit_training",
    "replace_lora_weights_loftq",
    "set_peft_model_state_dict",
    "shift_tokens_right",
]
