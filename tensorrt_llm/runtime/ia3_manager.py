import json
import os
import torch


class Ia3Config(object):

    def __init__(self,
                 hf_ia3_dir: str = None,
                 is_valid: bool = False,
                 ia3_target_modules: list = [],
                 hf_modules_to_trtllm_modules: dict = {}):
        self.hf_ia3_dir = hf_ia3_dir
        self.is_valid = is_valid
        self.hf_ia3_target_modules = ia3_target_modules
        self.ia3_target_modules = [
            hf_modules_to_trtllm_modules[m] for m in ia3_target_modules
        ]

    @classmethod
    def from_hf(cls, hf_ia3_dir, hf_modules_to_trtllm_modules):
        ia3_target_modules = {}
        adapter_config = None
        is_valid = True

        if os.path.exists(f"{hf_ia3_dir}/adapter_config.json"):
            with open(f"{hf_ia3_dir}/adapter_config.json") as f:
                adapter_config = json.load(f)
            ia3_target_modules = adapter_config["target_modules"]
        else:
            is_valid = False

        return cls(
            hf_ia3_dir=hf_ia3_dir,
            is_valid=is_valid,
            ia3_target_modules=ia3_target_modules,
            hf_modules_to_trtllm_modules=hf_modules_to_trtllm_modules
        )


class Ia3Manager(object):

    def __init__(self):
        self._ia3_uid_to_key = {}
        '''
        _ia3_weights_pointers_list:
        [
            {
               uid: 
               {
                   ia3_module_1: t
                   ia3_module_2: t
               }
            }, # layer_0
            {

            }, # layer_1
            ...
        ]

        '''
        self._ia3_weights = []
        self._ia3_weights_pointers_list = []
    
    def load_from_hf(self, model_dir, model_config):
        ia3_model = torch.load(f'{model_dir}/adapter_model.bin')
        
        ia3_target_modules = ['attn_v', 'attn_k', 'mlp_4h_to_h'] # model_config.ia3_target_modules
        
        for layer_idx in range(model_config.num_layers):

            self._ia3_weights_pointers_list.append({})
            self._ia3_weights_pointers_list[layer_idx].update({0: {}})
            
            prefix = "base_model.model.model.layers"
            for ia3_module in ia3_target_modules:
                if ia3_module == 'attn_k' or ia3_module == 'attn_v':
                    name = f"{prefix}.{layer_idx}.{ia3_module.replace('attn_', 'self_attn.')}_proj"
                elif ia3_module == 'mlp_4h_to_h':
                    name = f"{prefix}.{layer_idx}.mlp.down_proj"
                else:
                    raise ValueError(f'Unknown ia3 module: {ia3_module}')
                tensor = ia3_model[f'{name}.ia3_l'].cuda().contiguous()

                self._ia3_weights_pointers_list[layer_idx][0].update(
                    {ia3_module: tensor.data_ptr()}
                )
                self._ia3_weights.append(tensor)
                
    @property
    def ia3_weights(self):
        return self._ia3_weights

    @property
    def ia3_weights_pointers_list(self):
        return self._ia3_weights_pointers_list
