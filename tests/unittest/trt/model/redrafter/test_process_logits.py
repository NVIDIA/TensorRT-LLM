# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import torch
from parameterized import parameterized
from utils.util import create_session, run_session, set_input_shapes

import tensorrt_llm
import tensorrt_llm.models.redrafter
import tensorrt_llm.models.redrafter.redrafter_helper
from tensorrt_llm import Tensor
from tensorrt_llm.layers import SpecDecodingParams

REF_RESULTS = [
    {
        'probs':
        torch.tensor([[0., 0., 1., 0.]], device='cuda:0'),
        'drafter_input':
        torch.tensor([[0.0061, 0.2896]], device='cuda:0'),
        'num_accepted_tokens':
        torch.tensor([0], device='cuda:0', dtype=torch.int32),
        'accepted_beam_index':
        torch.tensor([0], device='cuda:0', dtype=torch.int32),
    },
    {
        'probs':
        torch.tensor([[0., 0., 0., 0.]], device='cuda:0'),
        'drafter_input':
        torch.tensor([[0.0061, 0.2896]], device='cuda:0'),
        'num_accepted_tokens':
        torch.tensor([0], device='cuda:0', dtype=torch.int32),
        'accepted_beam_index':
        torch.tensor([0], device='cuda:0', dtype=torch.int32),
    },
    {
        'probs':
        torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 1.],
                      [0., 0., 1., 0.], [0., 1., 0., 0.]],
                     device='cuda:0'),
        'drafter_input':
        torch.tensor([[0.0061, 0.2896], [0.6216, 0.3729], [0.1850, 0.8467],
                      [0.0706, 0.7531], [0.6967, 0.7016]],
                     device='cuda:0'),
        'num_accepted_tokens':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
        'accepted_beam_index':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
    },
    {
        'probs':
        torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
                      [0., 0., 0., 0.], [0., 0., 0., 0.]],
                     device='cuda:0'),
        'drafter_input':
        torch.tensor([[0.0061, 0.2896], [0.4299, 0.9747], [0.1328, 0.5547],
                      [0.4444, 0.1530], [0.5081, 0.7730]],
                     device='cuda:0'),
        'num_accepted_tokens':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
        'accepted_beam_index':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
    },
    {
        'probs':
        torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
                      [0., 0., 0., 0.], [0., 0., 0., 0.]],
                     device='cuda:0'),
        'drafter_input':
        torch.tensor([[0.0061, 0.2896], [0.6216, 0.3729], [0.1328, 0.5547],
                      [0.5722, 0.4428], [0.7922, 0.0827]],
                     device='cuda:0'),
        'num_accepted_tokens':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
        'accepted_beam_index':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
    },
    {
        'probs':
        torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 1.],
                      [0., 0., 1., 0.], [0., 0., 0., 0.]],
                     device='cuda:0'),
        'drafter_input':
        torch.tensor([[0.0061, 0.2896], [0.6216, 0.3729], [0.1850, 0.8467],
                      [0.0706, 0.7531], [0.6967, 0.7016]],
                     device='cuda:0'),
        'num_accepted_tokens':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
        'accepted_beam_index':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
    },
    {
        'probs':
        torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 1.],
                      [0., 0., 0., 0.], [0., 0., 0., 0.]],
                     device='cuda:0'),
        'drafter_input':
        torch.tensor([[0.0061, 0.2896], [0.6216, 0.3729], [0.1850, 0.8467],
                      [0.0706, 0.7531], [0.4207, 0.8064]],
                     device='cuda:0'),
        'num_accepted_tokens':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
        'accepted_beam_index':
        torch.tensor([0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32),
    },
]


class TestReDrafter(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')


########################################################################################################################

    @parameterized.expand([
        ([0], REF_RESULTS[0]),
        ([1], REF_RESULTS[1]),
        ([0, 0, 0, 0, 0], REF_RESULTS[2]),
        ([1, 1, 1, 1, 1], REF_RESULTS[3]),
        ([0, 1, 1, 1, 1], REF_RESULTS[4]),
        ([0, 0, 0, 0, 1], REF_RESULTS[5]),
        ([0, 0, 0, 1, 1], REF_RESULTS[6]),
    ])
    def test_process_logits(self,
                            rtypes=[0, 0, 0, 1, 1],
                            ref_res=REF_RESULTS[6]):
        # test data
        bs = len(rtypes)
        nb = 3
        bl = 4
        V = 4
        H = 2
        bs_gen = sum(rtypes)
        bs_ctx = bs - bs_gen
        greedy_search = True
        bs_gen4a = max(bs_gen, 1)

        old_device = torch.get_default_device()
        torch.set_default_device("cuda")
        torch.manual_seed(7)
        spec_decoding_generation_lengths = torch.randint(
            4, size=(bs_gen4a, ), dtype=torch.int32) + 4
        min_gen_token = 1 if bs_gen == 0 else min(
            spec_decoding_generation_lengths)
        S = sum(spec_decoding_generation_lengths)
        lm_logits = torch.rand((bs_ctx + S, V), dtype=torch.float32)
        hidden = torch.rand((bs_ctx + S, H), dtype=torch.float32)
        lm_logits *= 15.71
        draft_probs = torch.rand(
            (bs_gen4a, nb, bl - 1, V), dtype=torch.float32) * 41.13
        draft_tokens = torch.randint(100,
                                     (bs_gen4a, nb, bl), dtype=torch.int32) % V
        draft_indices = torch.randint(100, (bs_gen4a, nb, bl),
                                      dtype=torch.int32) % min_gen_token
        trt_dtype = tensorrt_llm.str_dtype_to_trt('float32')
        device_request_types = torch.tensor(rtypes, dtype=torch.int32)
        inverted_temperature = torch.ones_like(device_request_types,
                                               dtype=torch.float32)
        torch.manual_seed(11)
        rand_data = torch.rand((bs_gen4a, nb, bl - 1), dtype=torch.float32)

        # construct trt network
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            lm_logits_t = Tensor(name='l', shape=[-1, V], dtype=trt_dtype)
            hidden_t = Tensor(name='h', shape=[-1, H], dtype=trt_dtype)
            draft_probs_t = Tensor(
                name='dp',
                shape=[-1, nb, bl - 1, V],  #draft_probs.shape,
                dtype=trt_dtype)
            draft_tokens_t = Tensor(
                name='dt',
                shape=[-1, nb, bl],  #draft_tokens.shape,
                dtype=tensorrt_llm.torch_dtype_to_trt(draft_tokens.dtype))
            draft_indices_t = Tensor(
                name='di',
                shape=[-1, nb, bl],  #draft_indices.shape,
                dtype=tensorrt_llm.torch_dtype_to_trt(draft_indices.dtype))
            device_request_types_t = Tensor(
                name='drt',
                shape=[-1],  #device_request_types.shape,
                dtype=tensorrt_llm.torch_dtype_to_trt(
                    device_request_types.dtype))
            spec_decoding_generation_lengths_t = Tensor(
                name='sdgl',
                shape=[-1],  #spec_decoding_generation_lengths.shape,
                dtype=tensorrt_llm.torch_dtype_to_trt(
                    spec_decoding_generation_lengths.dtype))
            inverted_temperature_t = Tensor(
                name='invt',
                shape=[-1],  #spec_decoding_generation_lengths.shape,
                dtype=tensorrt_llm.torch_dtype_to_trt(
                    inverted_temperature.dtype))
            rand_data_t = Tensor(
                name='rand_data',
                shape=[-1, nb, bl - 1],  #rand_data.shape,
                dtype=tensorrt_llm.torch_dtype_to_trt(rand_data.dtype))

            kwargs = {}
            kwargs['num_beams'] = nb
            kwargs['beam_length'] = bl
            kwargs['greedy_search'] = greedy_search
            kwargs['device_request_types'] = device_request_types_t
            kwargs['spec_decoding_params'] = SpecDecodingParams(
                spec_decoding_generation_lengths=
                spec_decoding_generation_lengths_t)
            kwargs['draft_probs'] = draft_probs_t
            kwargs['draft_tokens'] = draft_tokens_t
            kwargs['draft_indices'] = draft_indices_t
            kwargs['rand_data_validation'] = rand_data_t
            kwargs['redrafter_inverted_temperature'] = inverted_temperature_t
            outputs = tensorrt_llm.models.redrafter.redrafter_helper._process_logits_and_hidden_states(
                None, lm_logits_t, hidden_t, kwargs)
            outputs[0].mark_output('probs')
            outputs[1].mark_output('drafter_input')
            outputs[2].mark_output('num_accepted_tokens')
            outputs[3].mark_output('accepted_beam_index')
            # save onnx
            # model_path = 'process.onnx'
            # to_onnx(net.trt_network, model_path)

        # trt run
        # needs profile for dynamic shape
        profile = builder.trt_builder.create_optimization_profile()
        set_input_shapes(profile, draft_probs_t, [0, nb, bl - 1, V],
                         [16, nb, bl - 1, V], [32, nb, bl - 1, V])
        set_input_shapes(profile, draft_tokens_t, [0, nb, bl], [16, nb, bl],
                         [32, nb, bl])
        set_input_shapes(profile, draft_indices_t, [0, nb, bl], [16, nb, bl],
                         [32, nb, bl])
        set_input_shapes(profile, device_request_types_t, [1], [16], [32])
        set_input_shapes(profile, spec_decoding_generation_lengths_t, [0], [16],
                         [32])
        set_input_shapes(profile, inverted_temperature_t, [1], [16], [32])
        set_input_shapes(profile, rand_data_t, [0, nb, bl - 1],
                         [16, nb, bl - 1], [32, nb, bl - 1])
        set_input_shapes(profile, lm_logits_t, [1, V], [256, V], [512, V])
        set_input_shapes(profile, hidden_t, [1, H], [256, H], [512, H])
        session = create_session(builder,
                                 network,
                                 precision='float32',
                                 optimization_profiles=[profile])
        inputs = {
            'l': lm_logits,
            'h': hidden,
            'dp': draft_probs,
            'dt': draft_tokens,
            'di': draft_indices,
            'drt': device_request_types,
            'sdgl': spec_decoding_generation_lengths,
            'invt': inverted_temperature,
            'rand_data': rand_data,
        }
        override_shapes = {
            'dp': (bs_gen, nb, bl - 1, V),
            'dt': (bs_gen, nb, bl),
            'di': (bs_gen, nb, bl),
            'sdgl': (bs_gen, ),
            'rand_data': (bs_gen, nb, bl - 1)
        }
        outputs = run_session(session, inputs, override_shapes=override_shapes)
        torch.testing.assert_close(ref_res['probs'],
                                   outputs['probs'],
                                   atol=0.01,
                                   rtol=0.1)
        torch.testing.assert_close(ref_res['drafter_input'],
                                   outputs['drafter_input'],
                                   atol=0.01,
                                   rtol=0.1)
        torch.testing.assert_close(ref_res['num_accepted_tokens'],
                                   outputs['num_accepted_tokens'],
                                   atol=0,
                                   rtol=0)
        torch.testing.assert_close(ref_res['accepted_beam_index'],
                                   outputs['accepted_beam_index'],
                                   atol=0,
                                   rtol=0)
        torch.set_default_device(old_device)
        return
