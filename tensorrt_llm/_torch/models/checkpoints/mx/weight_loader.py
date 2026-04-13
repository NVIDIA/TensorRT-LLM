# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelExpress P2P weight loader — receives weights via RDMA from an MX source."""

import logging
import os
from typing import Any, Dict, Union

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader,
    ConsumableWeightsDict,
)
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_weight_loader
from tensorrt_llm.mapping import Mapping

logger = logging.getLogger(__name__)


@register_checkpoint_weight_loader("MX")
class MxWeightLoader(BaseWeightLoader):

    def __init__(self, mx_server_url: str | None = None):
        self._mx_server_url = mx_server_url
        self._received_via_rdma = False

    @property
    def received_via_rdma(self) -> bool:
        return self._received_via_rdma

    def load_weights(
        self,
        checkpoint_dir: str,
        mapping: Mapping,
        **kwargs,
    ) -> Union[Dict[str, Any], ConsumableWeightsDict]:
        mx_server = (
            self._mx_server_url
            or os.environ.get("MODEL_EXPRESS_URL")
            or os.environ.get("MX_SERVER_ADDRESS", "localhost:8001")
        )

        try:
            from modelexpress.client import MxClient
        except ImportError as err:
            raise ImportError(
                "checkpoint_format='MX' requires the 'modelexpress' package. "
                "Install with: pip install modelexpress"
            ) from err

        probe_timeout = int(os.environ.get("MX_SOURCE_PROBE_TIMEOUT", "30"))
        try:
            mx_client = MxClient(server_url=mx_server)
            try:
                resp = mx_client.list_sources()
                has_sources = len(resp.instances) > 0
            finally:
                mx_client.close()
        except Exception:
            has_sources = False

        if has_sources:
            logger.info("MX source found — loading weights via RDMA")
            from modelexpress.trtllm_live_transfer import MxLiveWeightLoader
            live_loader = MxLiveWeightLoader(mx_server=mx_server)
            model = kwargs.get("model")
            result = live_loader.load_weights(
                checkpoint_dir, mapping=mapping, model=model
            )
            self._received_via_rdma = True
            os.environ["MODEL_EXPRESS_TARGET"] = "1"
            return result
        else:
            logger.info(
                "No MX source found — loading from disk, will publish as source"
            )
            os.environ["MODEL_EXPRESS_URL"] = mx_server
            from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import (
                HfWeightLoader,
            )
            return HfWeightLoader().load_weights(checkpoint_dir, mapping=mapping)

    def cleanup(self) -> None:
        pass
