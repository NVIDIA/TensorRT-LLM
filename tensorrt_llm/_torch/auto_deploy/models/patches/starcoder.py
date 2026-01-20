from transformers.models.starcoder2.configuration_starcoder2 import Starcoder2Config

# TODO: Remove this patch after TRT-LLM upgrades to the HF transformers version >= 4.57
Starcoder2Config.base_model_tp_plan["layers.*.mlp.c_proj"] = "rowwise"
