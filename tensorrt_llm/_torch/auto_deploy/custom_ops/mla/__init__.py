# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""MLA (Multi-head Latent Attention) custom ops.

This module provides various MLA implementations and backends:
- torch_mla: PyTorch reference MLA implementation
- torch_backend_mla: PyTorch-based MLA backend
- flashinfer_mla: FlashInfer-based optimized MLA
- flashinfer_trtllm_mla: FlashInfer TRTLLM-gen MLA (Blackwell Path 2)
- triton_mla: Triton-based MLA implementation
- trtllm_mla: TRT-LLM thop.attention-based MLA (requires tensorrt_llm)
"""
