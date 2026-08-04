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

import numpy as np

# isort: off
import torch
# isort: on
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    def ref_categorical_sample(self, probs: torch.Tensor):
        probs = probs / probs.sum(-1, keepdim=True)
        rand_data = torch.rand(probs.shape[0],
                               dtype=probs.dtype,
                               device=probs.device)
        cum_probs = probs.cumsum(-1)
        samples = (cum_probs >= rand_data.unsqueeze(1)).int().argmax(dim=-1)
        # print(samples)
        return samples

    # @unittest.skip("")
    def test_ref_sample(self):
        bs = 2
        nbins = 10
        probs = torch.rand((bs, nbins), dtype=torch.float32)
        scaled_probs = probs / probs.sum(-1, keepdim=True)
        print(scaled_probs)
        samples = []
        reps = 20000
        for _ in range(reps):
            samples.append(self.ref_categorical_sample(probs))
        samples = torch.stack(samples).float()
        # print(samples[:, 0], samples[:, 1])
        hist = []
        bins = torch.arange(nbins + 1).float()
        for i in range(bs):
            h = torch.histogram(samples[:, i], bins=bins).hist
            h = h / h.sum(-1)
            hist.append(h)
        np.testing.assert_allclose(torch.stack(hist), scaled_probs, atol=1e-2)
        return

    def test_sample(self):
        # test data
        bs = 2
        nbins = 10
        probs = torch.rand((bs, nbins), dtype=torch.float32)
        scaled_probs = probs / probs.sum(-1, keepdim=True)
        print(scaled_probs)

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=probs.shape,
                       dtype=tensorrt_llm.torch_dtype_to_trt(probs.dtype))
            # NOTE: we need rand() here since TRT rand() produces same numbers
            rand_data_t = Tensor(name='rand_data',
                                 shape=(bs, ),
                                 dtype=tensorrt_llm.torch_dtype_to_trt(
                                     torch.float32))

            outputs = tensorrt_llm.functional.categorical_sample(x, rand_data_t)
            outputs.trt_tensor.name = 'output'
            network.mark_output(outputs.trt_tensor)
            # save onnx
            # model_path = 'sample.onnx'
            # to_onnx(net.trt_network, model_path)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        samples = []
        nreps = 20000
        with TrtRunner(build_engine) as runner:
            for _ in range(nreps):
                # NOTE: we need rand() here since TRT rand() produces same numbers
                rand_data = torch.rand((bs, ), dtype=torch.float32)
                outputs = runner.infer(feed_dict={
                    'x': probs.numpy(),
                    'rand_data': rand_data.numpy(),
                })
                # print(outputs)
                samples.append(torch.tensor(outputs['output']))
        # assert False, "PARTIAL"
        samples = torch.stack(samples).float()
        print(samples)
        hist = []
        bins = torch.arange(nbins + 1).float()
        for i in range(bs):
            h = torch.histogram(samples[:, i], bins=bins).hist
            h = h / h.sum(-1)
            hist.append(h)
        print(hist)

        # compare diff
        np.testing.assert_allclose(torch.stack(hist), scaled_probs, atol=1e-2)
        # assert False, "FORCED"
        return
