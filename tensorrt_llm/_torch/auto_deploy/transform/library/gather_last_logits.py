# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Transform to fuse last-token gather+concat into the model graph.

This moves the Python torch.cat operation into the traced graph, enabling
CUDA graph capture of the gather operation.

Before (Python, outside graph):
    logits = model(...)                    # [batch, seq_len, vocab] or [1, total_tokens, vocab]
    logits_flat = torch.cat([             # Python loop + D2D copy
        ls_one_seq[-last_only:] for ls_one_seq, last_only in zip(logits, last_logit_only)
    ], dim=0)

After (inside graph):
    logits = model(...)                    # [batch, seq_len, vocab] or [1, total_tokens, vocab]
    logits_gathered = gather_last_logits(logits, seq_len_buffer)  # [max_batch, vocab]
    # ^ Fixed shape for CUDA graph compatibility
"""

from typing import Optional, Tuple

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...utils.logger import ad_logger
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


# Register the custom op for gather_last_logits
@torch.library.custom_op("auto_deploy::gather_last_logits", mutates_args=())
def gather_last_logits(
    logits: torch.Tensor,
    seq_len: torch.Tensor,
) -> torch.Tensor:
    """Gather last token logits from packed sequences.

    Args:
        logits: Logits tensor [batch, seq_len, vocab] or [1, total_tokens, vocab]
        seq_len: Sequence lengths [max_batch_size], only first N entries are valid

    Returns:
        Last token logits [max_batch_size, vocab] - fixed shape for CUDA graph
    """
    max_batch_size = seq_len.shape[0]
    vocab_size = logits.shape[-1]

    # Handle different input formats
    if logits.dim() == 3:
        if logits.shape[0] == 1:
            # Packed format (context): [1, total_tokens, vocab] -> [total_tokens, vocab]
            logits = logits.squeeze(0)
        else:
            # Generate format: [batch, 1, vocab] -> [batch, vocab]
            # Extract last token (which is the only token in generate mode)
            batch_size = logits.shape[0]
            logits = logits[:, -1, :]  # [batch, vocab]
            # Pad to max_batch_size for fixed output shape
            if batch_size < max_batch_size:
                padding = torch.zeros(
                    max_batch_size - batch_size,
                    vocab_size,
                    dtype=logits.dtype,
                    device=logits.device,
                )
                logits = torch.cat([logits, padding], dim=0)
            return logits

    # Packed format: [total_tokens, vocab]
    # Compute last token indices: cumsum(seq_lens) - 1
    last_token_indices = torch.cumsum(seq_len.long(), dim=0) - 1

    # Clamp indices to valid range for CUDA graph compatibility
    # During graph capture for smaller batch sizes, logits may have fewer rows
    max_idx = logits.shape[0] - 1
    last_token_indices = torch.clamp(last_token_indices, min=0, max=max_idx)

    # Gather last tokens: [max_batch, vocab]
    gathered = logits[last_token_indices]
    return gathered


@gather_last_logits.register_fake
def gather_last_logits_fake(
    logits: torch.Tensor,
    seq_len: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation for tracing - returns shape [max_batch_size, vocab]."""
    max_batch_size = seq_len.shape[0]
    vocab_size = logits.shape[-1]
    return logits.new_empty(max_batch_size, vocab_size)


def _get_model_device(gm: GraphModule) -> torch.device:
    """Get the device of the model from its parameters/buffers."""
    for param in gm.parameters():
        return param.device
    for buffer in gm.buffers():
        return buffer.device
    return torch.device("cuda")


def _register_seq_len_buffer(gm: GraphModule, max_batch_size: int, insert_before: Node) -> Node:
    """Register a seq_len buffer on the GraphModule.

    Using a buffer avoids changing input signature (which breaks CUDA graph capture).
    The buffer stores sequence lengths and gets updated at runtime via .copy_().
    """
    buffer_name = "_gather_seq_len_buffer"
    if not hasattr(gm, buffer_name):
        device = _get_model_device(gm)
        # Initialize with ones (each seq has at least 1 token)
        buffer = torch.ones(max_batch_size, dtype=torch.int32, device=device)
        gm.register_buffer(buffer_name, buffer)

    # Create a node to access the buffer - insert BEFORE where it's used
    with gm.graph.inserting_before(insert_before):
        buffer_node = gm.graph.get_attr(buffer_name)

    return buffer_node


def _find_output_node(gm: GraphModule) -> Optional[Node]:
    """Find the output node of the graph."""
    for node in gm.graph.nodes:
        if node.op == "output":
            return node
    return None


class GatherLastLogitsConfig(TransformConfig):
    """Configuration for GatherLastLogits transform."""

    enabled: bool = Field(
        default=False,  # Disabled by default, opt-in
        description="Whether to enable the gather-last-logits optimization.",
    )
    max_batch_size: int = Field(
        default=1024, description="Maximum batch size for the seq_len buffer."
    )


@TransformRegistry.register("gather_last_logits")
class GatherLastLogitsTransform(BaseTransform):
    """Transform to fuse last-token gather+concat into the model graph.

    This transform inserts a gather operation after the model's logits output
    to select only the last token logits for each sequence. The output is always
    [max_batch_size, vocab] for CUDA graph compatibility.

    Benefits:
    - Eliminates Python loop overhead
    - Enables CUDA graph capture of the gather
    - Reduces D2D copy overhead by doing it inside the graph

    NOTE: This transform assumes all sequences only need their last token logits.
    If you need full context logits, do not enable this transform.
    """

    config: GatherLastLogitsConfig

    @classmethod
    def get_config_class(cls):
        return GatherLastLogitsConfig

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Return early if disabled
        if not self.config.enabled:
            ad_logger.debug("GatherLastLogits transform is disabled (opt-in required)")
            return gm, TransformInfo(skipped=True, num_matches=0)

        ad_logger.info("Applying GatherLastLogits transform...")

        # Find the output node
        output_node = _find_output_node(gm)
        if output_node is None:
            ad_logger.warning("Could not find output node, skipping GatherLastLogits transform")
            return gm, TransformInfo(skipped=True, num_matches=0)

        # Get the logits tensor (first output argument)
        if not output_node.args or not output_node.args[0]:
            ad_logger.warning("Output node has no args, skipping GatherLastLogits transform")
            return gm, TransformInfo(skipped=True, num_matches=0)

        output_args = output_node.args[0]
        if isinstance(output_args, (list, tuple)):
            logits_node = output_args[0] if output_args else None
        else:
            logits_node = output_args

        if logits_node is None:
            ad_logger.warning("Could not find logits node, skipping GatherLastLogits transform")
            return gm, TransformInfo(skipped=True, num_matches=0)

        ad_logger.info(f"Found logits node: {logits_node.name}")

        # Register seq_len buffer and get node to access it
        seq_len_node = _register_seq_len_buffer(
            gm, max_batch_size=self.config.max_batch_size, insert_before=output_node
        )
        ad_logger.info(
            f"Registered seq_len buffer with max_batch_size={self.config.max_batch_size}"
        )

        # Insert gather operation before output
        with gm.graph.inserting_before(output_node):
            gather_node = gm.graph.call_function(
                torch.ops.auto_deploy.gather_last_logits,
                args=(logits_node, seq_len_node),
            )

        # Update output to use gathered logits
        if isinstance(output_args, (list, tuple)):
            new_output_args = (gather_node,) + tuple(output_args[1:])
        else:
            new_output_args = gather_node
        output_node.args = (new_output_args,)

        # Mark the model so executor knows gather is already done
        gm._gather_last_logits_applied = True

        # Recompile
        gm.graph.lint()
        gm.recompile()

        ad_logger.info(f"Successfully inserted gather_last_logits after: {logits_node.name}")
        return gm, TransformInfo(skipped=False, num_matches=1)
