# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger

from .modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)


@dataclass(frozen=True)
class MultimodalEncoderOutput:
    """Output produced by a model-owned multimodal encoder hook.

    This mixin currently supports a strict primary tensor output:
    - `embeddings` contains all multimodal embedding rows for the supplied
      `multimodal_params`.
    - Rows are concatenated in the same order as `multimodal_params`.
    - Per-request row counts match `total_embeds_in_request` from runtime
      metadata when that metadata is available.
    - Special multimodal tokens occupy token positions but do not have rows in
      this tensor.
    """

    embeddings: torch.Tensor
    cacheable: bool = True


@dataclass(frozen=True)
class PreparedLlmInputs:
    """Prepared inputs returned by `MultimodalModelMixin`."""

    input_ids: Optional[torch.Tensor]
    inputs_embeds: Optional[torch.Tensor]
    extra_embeds: Sequence[torch.Tensor] = ()


class MultimodalModelMixin:
    """Template-method mixin for PyTorch multimodal causal LM models.

    Concrete model forwards can call `prepare_multimodal_inputs` while
    keeping their explicit language-model delegation. A future optional
    mixin-owned forward can build on the same template method.
    """

    def encode_multimodal_inputs(
        self,
        multimodal_params: Sequence[MultimodalParams],
        **encoder_kwargs: Any,
    ) -> MultimodalEncoderOutput:
        """Run model-specific multimodal encoder work."""
        raise NotImplementedError

    @property
    def multimodal_token_ids(self) -> Optional[Sequence[int] | torch.Tensor]:
        """Return placeholder token ids in `input_ids` replaced by MM embeds.

        These are sentinel token positions whose text embeddings are replaced
        by multimodal embeddings. Return `None` to use the out-of-vocabulary
        sentinel behavior in `fuse_input_embeds`.
        """
        raise NotImplementedError

    @property
    def text_embedding_layer(self):
        """Return the token embedding layer used by `fuse_input_embeds`."""
        raise NotImplementedError

    def get_multimodal_encoder_kwargs(
        self,
        *,
        input_ids: torch.Tensor,
        multimodal_params: Sequence[MultimodalParams],
        **forward_kwargs: Any,
    ) -> dict[str, Any]:
        """Optional model hook for encoder-specific kwargs."""
        return {}

    def select_multimodal_params(
        self,
        multimodal_params: Sequence[MultimodalParams],
        num_context_requests: int,
    ) -> Sequence[MultimodalParams]:
        """Select the params that participate in multimodal encoder work.

        Returns the context-slice params with multimodal content. Helpers below
        this method (`get_multimodal_embeddings`, `find_input_mm_embeds`,
        `fuse_input_embeds`) operate on the returned list and therefore see
        only `has_content()` params. Models overriding this hook must
        preserve that invariant.
        """
        return [
            param for param in list(multimodal_params)[:num_context_requests] if param.has_content()
        ]

    def after_full_multimodal_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        multimodal_params: Sequence[MultimodalParams],
        encoder_output: MultimodalEncoderOutput,
        **forward_kwargs: Any,
    ) -> tuple[torch.Tensor, MultimodalEncoderOutput]:
        """Optional hook before active chunk rows are selected.

        Runs after cache lookup or encoder execution has produced full
        per-request multimodal embeddings, but before the mixin selects rows
        active in the current forward chunk.
        """
        return input_ids, encoder_output

    def after_active_multimodal_embeddings(
        self,
        *,
        active_embeddings: list[torch.Tensor],
        multimodal_params: Sequence[MultimodalParams],
        **forward_kwargs: Any,
    ) -> tuple[list[torch.Tensor], Sequence[torch.Tensor]]:
        """Optional hook after active chunk rows are selected and before fusion.

        Models can transform or split the active multimodal embeddings here
        and return additional embedding tensors to fuse alongside the primary
        multimodal embeddings.
        """
        # Future Qwen3-VL migration can split packed deepstack features here
        # and return them as extra embeds without changing the base flow.
        return active_embeddings, ()

    def prepare_multimodal_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor],
        multimodal_params: Optional[Sequence[MultimodalParams]],
        num_context_requests: int,
        **forward_kwargs: Any,
    ) -> PreparedLlmInputs:
        """Prepare multimodal inputs for a concrete model forward.

        This method owns the common framework sequence around a model-specific
        encoder hook: retrieve/cache full request embeddings, select active
        chunk rows, run optional model hooks, and fuse rows into text embeds.
        """
        context_params = list(
            self.select_multimodal_params(
                multimodal_params or [],
                num_context_requests,
            )
        )
        if not context_params:
            return PreparedLlmInputs(input_ids=input_ids, inputs_embeds=None)

        encoder_kwargs = self.get_multimodal_encoder_kwargs(
            input_ids=input_ids,
            multimodal_params=context_params,
            **forward_kwargs,
        )
        full_output = self._get_or_encode_multimodal_embeddings(
            context_params,
            **encoder_kwargs,
        )

        input_ids, full_output = self.after_full_multimodal_embeddings(
            input_ids=input_ids,
            multimodal_params=context_params,
            encoder_output=full_output,
            **forward_kwargs,
        )

        active_embeddings = self._find_active_multimodal_embeddings(
            [full_output.embeddings],
            input_ids=input_ids,
            positions=positions,
            multimodal_params=context_params,
        )
        active_embeddings, extra_embeds = self.after_active_multimodal_embeddings(
            active_embeddings=active_embeddings,
            multimodal_params=context_params,
            **forward_kwargs,
        )

        fused_input_ids, inputs_embeds, fused_extra_embeds = self._fuse_multimodal_embeddings(
            input_ids=input_ids,
            multimodal_embeddings=active_embeddings,
            mm_token_ids=self.multimodal_token_ids,
            embedding_layer=self.text_embedding_layer,
            extra_embeds=extra_embeds,
            # `text_token_indices` / `mm_token_indices` are pre-computed by the
            # executor (see model_engine._prepare_inputs) and must reach
            # `fuse_input_embeds` to (a) preserve the active-chunk subset
            # contract when MM rows are a subset of visible MM tokens and
            # (b) avoid the torch.where host sync inside
            # `filter_mm_token_from_input_ids`.
            text_token_indices=forward_kwargs.get("text_token_indices"),
            mm_token_indices=forward_kwargs.get("mm_token_indices"),
        )
        return PreparedLlmInputs(
            input_ids=fused_input_ids,
            inputs_embeds=inputs_embeds,
            extra_embeds=fused_extra_embeds,
        )

    def _get_or_encode_multimodal_embeddings(
        self,
        multimodal_params: Sequence[MultimodalParams],
        **encoder_kwargs: Any,
    ) -> MultimodalEncoderOutput:
        """Return cached multimodal embeddings or run the encoder for misses.

        Delegates cache lookup and gather behavior to
        `get_multimodal_embeddings`, then validates the single primary tensor
        contract for both encoded and cached-only paths.
        """

        def encoder_forward_fn(params: list[MultimodalParams], **kwargs: Any) -> list[torch.Tensor]:
            encoder_output = self.encode_multimodal_inputs(params, **kwargs)
            if not isinstance(encoder_output, MultimodalEncoderOutput):
                raise TypeError("encode_multimodal_inputs must return MultimodalEncoderOutput.")
            if not isinstance(encoder_output.embeddings, torch.Tensor):
                raise TypeError("MultimodalEncoderOutput.embeddings must be a torch.Tensor.")
            if not encoder_output.cacheable:
                raise ValueError("MultimodalModelMixin requires cacheable encoder outputs.")
            return [encoder_output.embeddings]

        embeddings = get_multimodal_embeddings(
            encoder_forward_fn=encoder_forward_fn,
            multimodal_params=list(multimodal_params),
            encoder_kwargs=encoder_kwargs,
        )
        primary = self._require_primary_embedding(embeddings)
        # Validate post-gather so cached-only paths (KV reuse, all-cached chunked
        # prefill) are also checked, not just paths that ran the encoder.
        self._validate_primary_embedding_rows(primary, multimodal_params)
        return MultimodalEncoderOutput(embeddings=primary, cacheable=True)

    def _find_active_multimodal_embeddings(
        self,
        multimodal_embeddings: list[torch.Tensor],
        *,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor],
        multimodal_params: Sequence[MultimodalParams],
    ) -> list[torch.Tensor]:
        """Named internal stage for selecting active chunk multimodal rows.

        This initial template stage currently delegates to
        `find_input_mm_embeds`. Model-specific behavior around slicing should
        use `after_full_multimodal_embeddings` or
        `after_active_multimodal_embeddings` so the common mixin sequence stays
        centralized.
        """
        return find_input_mm_embeds(multimodal_embeddings, list(multimodal_params))

    def _fuse_multimodal_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor],
        mm_token_ids: Optional[Sequence[int] | torch.Tensor],
        embedding_layer,
        extra_embeds: Sequence[torch.Tensor],
        text_token_indices: Optional[torch.Tensor] = None,
        mm_token_indices: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Sequence[torch.Tensor]]:
        """Thin adapter over `fuse_input_embeds`.

        The framework does not forward `prepare_multimodal_inputs` kwargs
        into `fuse_input_embeds`; only inputs the helper actually consumes
        are surfaced here. Models needing to bypass token filtering should
        pass pre-computed `text_token_indices`/`mm_token_indices`.
        """
        if mm_token_ids is not None and not isinstance(mm_token_ids, torch.Tensor):
            mm_token_ids = torch.tensor(
                list(mm_token_ids), dtype=input_ids.dtype, device=input_ids.device
            )

        result = fuse_input_embeds(
            embedding_layer=embedding_layer,
            input_ids=input_ids,
            mm_embeds=multimodal_embeddings,
            mm_token_ids=mm_token_ids,
            text_token_indices=text_token_indices,
            mm_token_indices=mm_token_indices,
            extra_embeds=list(extra_embeds) if extra_embeds else None,
        )
        if len(result) == 3:
            fused_input_ids, inputs_embeds, fused_extra_embeds = result
            return fused_input_ids, inputs_embeds, fused_extra_embeds or ()

        fused_input_ids, inputs_embeds = result
        return fused_input_ids, inputs_embeds, ()

    @staticmethod
    def _require_primary_embedding(embeddings: list[torch.Tensor]) -> torch.Tensor:
        if len(embeddings) != 1:
            raise ValueError(
                "MultimodalModelMixin requires a single primary embedding tensor, "
                f"got {len(embeddings)} tensors."
            )
        return embeddings[0]

    @staticmethod
    def _validate_primary_embedding_rows(
        primary: torch.Tensor,
        multimodal_params: Sequence[MultimodalParams],
    ) -> None:
        """Validate gathered primary embedding row count against runtime metadata.

        Skipped if any param lacks `multimodal_runtime.total_embeds_in_request`, since the contract
        cannot be evaluated without complete metadata.
        """
        expected_rows = 0
        for param in multimodal_params:
            runtime = param.multimodal_runtime
            if runtime is None or runtime.total_embeds_in_request is None:
                logger.debug(
                    "Skipping multimodal embedding row-count validation: "
                    "runtime metadata missing or incomplete for at least one param."
                )
                return
            expected_rows += runtime.total_embeds_in_request

        actual_rows = primary.shape[0]
        if actual_rows != expected_rows:
            raise ValueError(
                "Multimodal embedding row count mismatch: "
                f"expected {expected_rows}, got {actual_rows}."
            )
