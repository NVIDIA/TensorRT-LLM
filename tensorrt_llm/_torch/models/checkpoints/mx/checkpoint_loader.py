# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""ModelExpress checkpoint loader.

TensorRT-LLM owns model construction, compatibility identity, post-transform
qualification, and post-load lifecycle. ModelExpress owns source selection,
RDMA transfer, native fallback selection, publication, and transport cleanup
through its shared load-strategy chain.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader
from tensorrt_llm._torch.weight_sharing import SOURCE_IDENTITY_FORMAT_VERSION, SourceIdentity
from tensorrt_llm.mapping import Mapping


def _enable_mx_transfer_logging() -> None:
    """Enable ModelExpress INFO records for requested per-rank transfer logs."""
    if not os.environ.get("MX_TRANSFER_LOG_DIR"):
        return

    mx_logger = logging.getLogger("modelexpress")
    if mx_logger.getEffectiveLevel() > logging.INFO:
        mx_logger.setLevel(logging.INFO)


@register_checkpoint_loader("MX")
class MXCheckpointLoader(HfCheckpointLoader):
    """Load a TRT-LLM shard through ModelExpress, with native HF fallback."""

    def __init__(
        self,
        *,
        weight_loader: Optional[BaseWeightLoader] = None,
        weight_mapper: Optional[BaseWeightMapper] = None,
        config_loader: Optional[BaseConfigLoader] = None,
        mx_server_url: Optional[str] = None,
        model_name: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            weight_loader=weight_loader,
            weight_mapper=weight_mapper,
            config_loader=config_loader,
        )
        self._checkpoint_format = "MX"
        self._mx_server_url = mx_server_url
        self._model_name = str(model_name) if model_name is not None else None
        self._p2p_succeeded = False
        self._post_transform_weights_preloaded = False
        self._source_identity_compatible_for_last_load = False
        self._transform_protocol_version_for_last_load: Optional[int] = None
        self._local_source_identity: Optional[SourceIdentity] = None
        self._mx_loader = None

    @property
    def checkpoint_format(self) -> str:
        return "MX"

    @property
    def mx_server_url(self) -> Optional[str]:
        return self._mx_server_url

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def is_weights_preloaded(self) -> bool:
        return self._p2p_succeeded

    def is_post_transform_weights_preloaded(self) -> bool:
        return (
            self._p2p_succeeded
            and self._post_transform_weights_preloaded
            and self._source_identity_compatible_for_last_load
        )

    def load_weights(self, checkpoint_dir: str, mapping: Mapping, **kwargs) -> dict[str, Any]:
        """Load weights through ModelExpress's shared strategy chain.

        Args:
            checkpoint_dir: Hugging Face checkpoint used by native fallback.
            mapping: Distributed rank and parallelism mapping.
            **kwargs: TRT-LLM-owned load state. ``model``, ``model_config``,
                and ``source_identity`` enable the shared strategy path;
                ``load_config`` is forwarded to ModelExpress. Qualified
                post-transform reception additionally supplies
                ``allow_post_transform_weights``,
                ``prepare_post_transform_receiver``, and
                ``post_transform_protocol_version``.

        Returns:
            Native checkpoint weights on fallback, or an empty dictionary
            after ModelExpress writes the complete shard into ``model``.

        Raises:
            ImportError: ModelExpress is requested but unavailable.
            RuntimeError: Qualified reception lacks its structure,
                protocol, or current SourceIdentity ABI contract.
        """
        model = kwargs.pop("model", None)
        self._local_source_identity = kwargs.pop("source_identity", None)
        allow_post_transform_weights = kwargs.pop("allow_post_transform_weights", False)
        prepare_post_transform_receiver = kwargs.pop("prepare_post_transform_receiver", None)
        model_config = kwargs.pop("model_config", None)
        load_config = kwargs.pop("load_config", None)
        transform_protocol_version = kwargs.pop("post_transform_protocol_version", None)
        self._p2p_succeeded = False
        self._post_transform_weights_preloaded = False
        self._source_identity_compatible_for_last_load = False
        self._transform_protocol_version_for_last_load = None

        if (
            self._mx_server_url is None
            or model is None
            or self._local_source_identity is None
            or model_config is None
        ):
            return super().load_weights(
                checkpoint_dir,
                mapping=mapping,
                **kwargs,
            )

        _enable_mx_transfer_logging()

        try:
            from modelexpress.engines.trtllm import MxModelLoader
        except ImportError as exc:
            raise ImportError(
                "ModelExpress checkpoint loading was explicitly requested, "
                "but the ModelExpress client could not be imported. Install "
                'the MX dependencies with `pip install "tensorrt-llm[mx]"`, '
                "or select a different `checkpoint_format`."
            ) from exc

        if allow_post_transform_weights and prepare_post_transform_receiver is None:
            raise RuntimeError("Qualified MX loading requires receiver structure preparation")
        if allow_post_transform_weights and transform_protocol_version is None:
            raise RuntimeError("Qualified MX loading requires a transform protocol version")
        if allow_post_transform_weights and (
            self._local_source_identity.format_version != SOURCE_IDENTITY_FORMAT_VERSION
            or not self._local_source_identity.transform_abi_id
        ):
            raise RuntimeError(
                "Qualified MX loading requires the current TRT-LLM "
                "SourceIdentity format and a transform-layout ABI"
            )

        if self._mx_loader is not None:
            self._mx_loader.cleanup()

        self._mx_loader = MxModelLoader(
            model_config=model_config,
            load_config=load_config,
            checkpoint_loader=self,
            checkpoint_dir=checkpoint_dir,
            native_loader_kwargs=kwargs,
            mapping=mapping,
            source_identity=self._local_source_identity,
            prepare_post_transform_receiver=(
                prepare_post_transform_receiver
                if prepare_post_transform_receiver is not None
                else lambda _model: None
            ),
            transform_protocol_version=transform_protocol_version,
            p2p_enabled=allow_post_transform_weights,
            mx_server_url=self._mx_server_url,
        )
        weights = self._mx_loader.load_model(model)
        self._p2p_succeeded = self._mx_loader.p2p_succeeded
        self._transform_protocol_version_for_last_load = self._mx_loader.transform_protocol_version
        post_transform_compatible = (
            self._p2p_succeeded
            and transform_protocol_version is not None
            and self._transform_protocol_version_for_last_load == transform_protocol_version
            # The real ModelExpress TRT-LLM adapter serializes the complete
            # authoritative SourceIdentity into its discovery identity. RDMA
            # success therefore means the selected source matched format v3,
            # the transform-layout ABI, and the remaining TRT identity fields.
            and self._local_source_identity.format_version == SOURCE_IDENTITY_FORMAT_VERSION
            and bool(self._local_source_identity.transform_abi_id)
        )
        if self._p2p_succeeded and not post_transform_compatible:
            self._p2p_succeeded = False
            raise RuntimeError(
                "MX transferred weights without a compatible TRT-LLM "
                "transform protocol and SourceIdentity ABI"
            )
        self._post_transform_weights_preloaded = post_transform_compatible
        self._source_identity_compatible_for_last_load = post_transform_compatible
        return weights

    def publish_as_source(
        self,
        model,
        checkpoint_dir: Optional[str] = None,
        *,
        source_identity: Optional[SourceIdentity] = None,
    ) -> None:
        """Publish post-transform weights through the active MX load session."""
        if self._mx_loader is not None:
            self._mx_loader.publish_model(model)

    def post_load_publish(
        self,
        model,
        *,
        checkpoint_dir: str,
        weights_preloaded: bool = False,
        source_identity: Optional[SourceIdentity] = None,
    ) -> None:
        """Publish a native-loaded source after TRT-LLM post-load processing."""
        if weights_preloaded:
            return
        self.publish_as_source(
            model,
            checkpoint_dir=checkpoint_dir,
            source_identity=source_identity,
        )

    def cleanup(self) -> None:
        if self._mx_loader is not None:
            self._mx_loader.cleanup()
            self._mx_loader = None
        super().cleanup()
