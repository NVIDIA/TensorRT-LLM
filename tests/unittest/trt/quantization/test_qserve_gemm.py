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
import tensorrt as trt
import torch
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
from polygraphy.logger import G_LOGGER

from tensorrt_llm.quantization.quantize import (qserve_pack_reorder_per_channel,
                                                qserve_pack_reorder_per_group)

from . import _utils

G_LOGGER.severity = 0

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.quantization.functional import (qserve_gemm_per_channel,
                                                  qserve_gemm_per_group)


class TestQServeGemm(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def _case_qserve_gemm_per_group(self,
                                    m,
                                    n,
                                    k,
                                    group_size=128,
                                    seed=123456,
                                    dtype="float16"):
        torch.manual_seed(seed)

        # Initialize qact (int8 range: -128 to 127)
        qact = torch.randint(-128, 128, (m, k), dtype=torch.int8)

        # Initialize act_scales (float16)
        act_scales = torch.rand(
            m, 1, dtype=torch.float16) * 0.1  # small positive values

        # Initialize qweight (int4 range: 0 to 15, as per qserve_quantize_weight)
        qweight = torch.randint(0, 16, (n, k), dtype=torch.int8)

        # Initialize s1_scales (float16)
        s1_scales = torch.rand(n, 1, dtype=torch.float16) * 0.1

        # Initialize s2_scales (int)
        s2_scales = torch.randint(1, 16, (n, k // group_size), dtype=torch.int8)

        # Initialize s2_zeros (int)
        s2_szeros = torch.randint(0, 16, (n, k // group_size), dtype=torch.int8)

        s2_szeros = s2_scales * s2_szeros

        # Ground truth matmul
        ref = _utils.gt_qserve_gemm_per_group(qact, act_scales, qweight,
                                              s1_scales, s2_scales,
                                              s2_szeros).cpu().numpy()

        # Prepare data for QServe gemm kernel
        qweight, s1_scales, s2_scales, s2_szeros = qserve_pack_reorder_per_group(
            qweight, s1_scales, s2_scales, s2_szeros, group_size)
        # Create builder
        builder = tensorrt_llm.Builder()
        builder.strongly_typed = False  # Test need to run in weekly typed mode
        # Create empty network
        network = builder.create_network()
        # Allow SQ plugin of dtype type
        network.plugin_config.set_qserve_plugins("float16")
        with tensorrt_llm.net_guard(network):
            qact_trt = Tensor(
                name='qact',
                shape=qact.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))
            act_scales_trt = Tensor(
                name='act_scales',
                shape=act_scales.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
            qweight_trt = Tensor(
                name='qweight',
                shape=qweight.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))
            s1_scales_trt = Tensor(
                name='s1_scales',
                shape=s1_scales.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
            s2_scales_trt = Tensor(
                name='s2_scales',
                shape=s2_scales.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))
            s2_zeros_trt = Tensor(
                name='s2_szeros',
                shape=s2_szeros.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))

            output = qserve_gemm_per_group(qact_trt, act_scales_trt,
                                           qweight_trt, s1_scales_trt,
                                           s2_scales_trt, s2_zeros_trt)
            output.mark_output('output', dtype)

        engine = EngineFromNetwork(
            (builder.trt_builder, network.trt_network),
            config=CreateConfig(int8=True,
                                fp16=True,
                                memory_pool_limits={
                                    trt.MemoryPoolType.WORKSPACE:
                                    32 * 1024 * 1024
                                }))

        # Infer engine
        with TrtRunner(engine) as runner:
            outputs = runner.infer(
                feed_dict={
                    'qact': qact.numpy(),
                    'act_scales': act_scales.numpy(),
                    'qweight': qweight.numpy(),
                    's1_scales': s1_scales.numpy(),
                    's2_scales': s2_scales.numpy(),
                    's2_szeros': s2_szeros.numpy()
                })

        output = outputs['output']
        # Allow difference by one code point.
        self.assertTrue(np.allclose(output, ref, rtol=0, atol=np.spacing(ref)))

    def _case_qserve_gemm_per_channel(self,
                                      m,
                                      n,
                                      k,
                                      seed=123456,
                                      dtype="float16"):
        torch.manual_seed(seed)

        # Initialize qact (int8 range: -128 to 127)
        qact = torch.randint(-128, 128, (m, k), dtype=torch.int8)

        # Initialize act_scales (float16)
        act_scales = torch.rand(
            m, 1, dtype=torch.float16) * 0.1  # small positive values

        # Compute act_sums
        act_sums = torch.sum(qact.float() * act_scales.float(),
                             dim=1,
                             keepdim=True).half()

        # Initialize qweight (int4 range: 0 to 15, as per qserve_quantize_weight)
        qweight = torch.randint(0, 16, (n, k), dtype=torch.int8)

        # Initialize s1_scales (float16)
        s1_scales = torch.rand(n, 1, dtype=torch.float16) * 0.1

        # Initialize s1_szeros (float16)
        s1_szeros = torch.randint(1, 16, (n, 1))
        s1_szeros = s1_szeros.half() * s1_scales

        # Ground truth matmul
        ref = _utils.gt_qserve_gemm_per_channel(qact, act_scales, act_sums,
                                                qweight, s1_scales,
                                                s1_szeros).cpu().numpy()

        # Prepare data for QServe gemm kernel
        qweight, s1_scales, s1_szeros = qserve_pack_reorder_per_channel(
            qweight, s1_scales, s1_szeros)

        # Create builder
        builder = tensorrt_llm.Builder()
        builder.strongly_typed = False  # Test need to run in weekly typed mode
        # Create empty network
        network = builder.create_network()
        # Allow SQ plugin of dtype type
        network.plugin_config.set_qserve_plugins("float16")
        with tensorrt_llm.net_guard(network):
            qact_trt = Tensor(
                name='qact',
                shape=qact.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))
            act_scales_trt = Tensor(
                name='act_scales',
                shape=act_scales.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
            act_sums_trt = Tensor(
                name='act_sums',
                shape=act_scales.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
            qweight_trt = Tensor(
                name='qweight',
                shape=qweight.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("int8"))
            s1_scales_trt = Tensor(
                name='s1_scales',
                shape=s1_scales.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))
            s1_szeros_trt = Tensor(
                name='s1_szeros',
                shape=s1_szeros.shape,
                dtype=tensorrt_llm._utils.str_dtype_to_trt("float16"))

            output = qserve_gemm_per_channel(qact_trt, act_scales_trt,
                                             act_sums_trt, qweight_trt,
                                             s1_scales_trt, s1_szeros_trt)
            output.mark_output('output', dtype)

        engine = EngineFromNetwork(
            (builder.trt_builder, network.trt_network),
            config=CreateConfig(int8=True,
                                fp16=True,
                                memory_pool_limits={
                                    trt.MemoryPoolType.WORKSPACE:
                                    32 * 1024 * 1024
                                }))

        # Infer engine
        with TrtRunner(engine) as runner:
            outputs = runner.infer(
                feed_dict={
                    'qact': qact.numpy(),
                    'act_scales': act_scales.numpy(),
                    'act_sums': act_sums.numpy(),
                    'qweight': qweight.numpy(),
                    's1_scales': s1_scales.numpy(),
                    's1_szeros': s1_szeros.numpy(),
                })

        output = outputs['output']
        # Allow some difference.
        self.assertTrue(np.allclose(output, ref, rtol=1e-2, atol=0.25))

    def test_qserve_gemm_per_group(self, dtype='float16'):
        bs = 2
        inseq = 16
        hidden_size = 768

        # qkv_gemm
        self._case_qserve_gemm_per_group(bs * inseq,
                                         3 * hidden_size,
                                         hidden_size,
                                         dtype=dtype)

        # mlp_gemm_1
        self._case_qserve_gemm_per_group(bs * inseq,
                                         4 * hidden_size,
                                         hidden_size,
                                         dtype=dtype)

    def test_qserve_gemm_per_channel(self, dtype='float16'):
        bs = 2
        inseq = 16
        hidden_size = 768

        # qkv_gemm
        self._case_qserve_gemm_per_channel(bs * inseq,
                                           3 * hidden_size,
                                           hidden_size,
                                           dtype=dtype)

        # mlp_gemm_1
        self._case_qserve_gemm_per_channel(bs * inseq,
                                           4 * hidden_size,
                                           hidden_size,
                                           dtype=dtype)

    def test_qserve_gemm_per_group_no_plugin(self):
        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            # Gemm ootb should fail
            with self.assertRaisesRegex(
                    TypeError,
                    "QServe Quant GEMM is only supported with plugin"):
                qserve_gemm_per_group(None, None, None, None, None, None)

    def test_qserve_gemm_per_channel_no_plugin(self):
        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        network = builder.create_network()
        with tensorrt_llm.net_guard(network):
            # Gemm ootb should fail
            with self.assertRaisesRegex(
                    TypeError,
                    "QServe Quant GEMM is only supported with plugin"):
                qserve_gemm_per_channel(None, None, None, None, None, None)


if __name__ == '__main__':
    unittest.main()
