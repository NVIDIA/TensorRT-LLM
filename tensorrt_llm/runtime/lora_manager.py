import json

import numpy as np

from .._utils import _str_to_np_dict, fromfile, numpy_to_torch


class LoraManager(object):

    def __init__(self, model_dir, model_config):
        '''
        Load lora modules, could be move to client side
        '''
        self._model_config = model_config
        self._lora_uid_to_key = {}
        self._lora_uid_to_low_ranks = {}
        self._lora_weights = []
        self._lora_weights_pointers_list = [
        ]  # shape: [layer, lora_module_numbers, 2]

        with open(model_dir / "lora_weights.json", 'r') as f:
            config = json.load(f)
        lora_config = config['lora_config']
        for key in lora_config['lora_kqv_adapter']:
            self._lora_uid_to_key[lora_config['lora_kqv_adapter'][key]
                                  ['key']] = key

        for layer_idx in range(model_config.num_layers):
            self._lora_weights_pointers_list.append({})

            for uid, key in self._lora_uid_to_key.items():
                low_rank = int(lora_config['lora_kqv_adapter'][key]['low_rank'])
                self._lora_uid_to_low_ranks[lora_config['lora_kqv_adapter'][key]
                                            ['key']] = low_rank
                prefix = f"model.model.language_model.encoder.layers.{layer_idx}.self_attention.adapter_layer.lora_kqv_adapter.{key}"
                t_in = numpy_to_torch(
                    np.ascontiguousarray(
                        fromfile(model_dir, f'{prefix}.linear_in.weight.bin',
                                 [model_config.hidden_size, low_rank],
                                 _str_to_np_dict['bfloat16']).transpose(
                                     1, 0))).cuda()

                t_out = numpy_to_torch(
                    np.ascontiguousarray(
                        fromfile(model_dir, f'{prefix}.linear_out.weight.bin',
                                 [low_rank, model_config.hidden_size * 3],
                                 _str_to_np_dict['bfloat16']).transpose(
                                     1, 0))).cuda()

                self._lora_weights_pointers_list[layer_idx].update({
                    uid: [
                        t_in.contiguous().data_ptr(),
                        t_out.contiguous().data_ptr()
                    ]
                })

                self._lora_weights.append(t_in)
                self._lora_weights.append(t_out)

    def uid_to_key(self, uid: str):
        assert isinstance(uid, str)
        return self._lora_uid_to_key[uid]

    def uid_to_low_ranks(self, uid: str):
        assert isinstance(uid, str)
        return self._lora_uid_to_low_ranks[uid]

    @property
    def lora_weights(self):
        return self._lora_weights

    @property
    def lora_weights_pointers_list(self):
        return self._lora_weights_pointers_list
