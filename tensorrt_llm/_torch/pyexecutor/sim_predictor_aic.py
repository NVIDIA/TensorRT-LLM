# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""AIConfigurator-based batch time predictor for simulation mode."""

from typing import Optional

import numpy as np

from aiconfigurator.sdk import models as aic_models
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import (
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
)
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.perf_database import get_database

from tensorrt_llm.logger import logger

from .sim_predictor import InferTimePredictor, SimBatch


class AIConfiguratorPredictor(InferTimePredictor):
    """Predicts batch execution time using AIConfigurator's analytical model.

    Uses per-operation silicon performance tables to predict latency
    for prefill and decode batches based on model architecture, hardware,
    batch size, and sequence lengths.

    Args:
        model_path: HuggingFace model path (e.g. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0').
        device_name: AIC system name (e.g. 'h100_sxm').
        backend_version: TRT-LLM version for database lookup (e.g. '1.2.0rc5').
        database_path: Custom AIC systems/ directory. None uses bundled database.
        tp_size: Tensor parallel size.
        prefill_scale_factor: Multiplicative correction for prefill predictions.
        decode_scale_factor: Multiplicative correction for decode predictions.
    """

    def __init__(
        self,
        model_path: str,
        device_name: str,
        backend_version: str,
        database_path: Optional[str] = None,
        tp_size: int = 1,
        prefill_scale_factor: float = 1.0,
        decode_scale_factor: float = 1.0,
    ):
        self._prefill_scale = prefill_scale_factor
        self._decode_scale = decode_scale_factor

        model_config = ModelConfig(
            tp_size=tp_size,
            gemm_quant_mode=GEMMQuantMode.float16,
            kvcache_quant_mode=KVCacheQuantMode.float16,
            fmha_quant_mode=FMHAQuantMode.float16,
            comm_quant_mode=CommQuantMode.half,
        )

        systems_paths = [database_path] if database_path else None
        database = get_database(
            system=device_name,
            backend="trtllm",
            version=backend_version,
            systems_paths=systems_paths,
        )
        if database is None:
            raise ValueError(
                f"Failed to load AIC database for system={device_name}, "
                f"backend=trtllm, version={backend_version}. "
                f"Check device_name and backend_version are valid.")

        perf_model = aic_models.get_model(
            model_path=model_path,
            model_config=model_config,
            backend_name="trtllm",
        )
        backend = get_backend("trtllm")

        self._session = InferenceSession(
            model=perf_model, backend=backend, database=database)

        logger.info(
            "[AIConfiguratorPredictor] Initialized for %s on %s (trtllm/%s)",
            model_path, device_name, backend_version)

    def predict(self, batch: SimBatch) -> float:
        if not batch.requests:
            return 0.0

        if batch.is_prefill:
            return self._predict_prefill(batch)
        return self._predict_decode(batch)

    def _predict_prefill(self, batch: SimBatch) -> float:
        mean_past = float(np.mean(
            [r.past_kv_length for r in batch.requests]))
        mean_input = float(np.mean(
            [r.input_length for r in batch.requests]))
        isl = int(mean_past + mean_input)
        prefix = int(mean_past)

        runtime_config = RuntimeConfig(
            batch_size=len(batch.requests), isl=isl, prefix=prefix, osl=1)

        summary = self._session.run_static(runtime_config, mode="static_ctx")
        latency_dict = summary.get_context_latency_dict()
        total_ms = sum(latency_dict.values())

        if summary.check_oom():
            logger.warning("[AIConfiguratorPredictor] OOM detected for prefill "
                           "batch_size=%d isl=%d", len(batch.requests), isl)
            return -abs(total_ms) * self._prefill_scale / 1e3

        return total_ms * self._prefill_scale / 1e3

    def _predict_decode(self, batch: SimBatch) -> float:
        isl = int(np.mean([r.past_kv_length for r in batch.requests]))

        runtime_config = RuntimeConfig(
            batch_size=len(batch.requests), isl=isl, osl=2)

        summary = self._session.run_static(runtime_config, mode="static_gen")
        latency_dict = summary.get_generation_latency_dict()
        total_ms = sum(latency_dict.values())

        if summary.check_oom():
            logger.warning("[AIConfiguratorPredictor] OOM detected for decode "
                           "batch_size=%d isl=%d", len(batch.requests), isl)
            return -abs(total_ms) * self._decode_scale / 1e3

        return total_ms * self._decode_scale / 1e3
