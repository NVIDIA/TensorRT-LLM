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

"""Replace fusible subgraphs in MLIR IR with fused opaque ops.

After subgraph discovery and kernel generation, this module replaces each
subgraph's ops with a single ``AdOpaque`` op. The opaque op carries metadata
that the MLIR-to-FX converter uses to reconstruct a call to the generated
Triton kernel.
"""

from typing import Any, Callable, Dict

from xdsl.dialects.builtin import StringAttr

from ..dialect import AdOpaque
from .subgraph_discovery import FusibleSubgraph


def replace_subgraph_with_fused_op(
    subgraph: FusibleSubgraph,
    kernel_fn: Callable,
    sg_hash: str,
    metadata: Dict[str, Dict[str, Any]],
) -> None:
    """Replace a subgraph's ops in the MLIR block with a single fused AdOpaque.

    Args:
        subgraph: The fusible subgraph to replace.
        kernel_fn: The generated kernel callable (unused directly, but the
            corresponding torch.ops function is looked up by name).
        sg_hash: The subgraph hash used in the torch.ops registration name.
        metadata: The FXToMLIRConverter metadata side-table. A new entry is
            added for the fused op so MLIR-to-FX can reconstruct it.
    """
    import torch
    from xdsl.dialects.builtin import TensorType

    from ..dialect import mlir_to_torch_dtype

    node_key = f"mlir_fused_{sg_hash}"
    op_fn = getattr(torch.ops.auto_deploy, node_key)

    # Build the args template: each subgraph input maps to a positional operand
    args_template = tuple(("__mlir_operand__", i) for i in range(len(subgraph.inputs)))

    # Build FakeTensor "val" metadata so downstream shape propagation works.
    # The fused op returns a tuple of tensors matching the subgraph output shapes.
    # Dynamic dimensions (-1 in MLIR) are replaced with a placeholder size (2)
    # since shape_prop will recompute exact shapes later.
    fake_vals = []
    for out in subgraph.outputs:
        if isinstance(out.type, TensorType):
            shape = [s if s >= 0 else 2 for s in out.type.get_shape()]
            dtype = mlir_to_torch_dtype(out.type.element_type)
            fake_vals.append(torch.empty(shape, dtype=dtype, device="meta"))
    val_meta = tuple(fake_vals) if len(fake_vals) > 1 else fake_vals[0] if fake_vals else None

    # Store synthetic metadata for the MLIR→FX converter
    metadata[node_key] = {
        "_original_target": op_fn,
        "_args_template": args_template,
        "_kwargs_template": {},
        "val": val_meta,
    }

    # Determine the MLIR result types from the subgraph outputs
    result_types = [out.type for out in subgraph.outputs]

    # Build the fused AdOpaque op
    fused_op = AdOpaque.build(
        operands=[list(subgraph.inputs)],
        attributes={
            "op_name": StringAttr(node_key),
            "node_key": StringAttr(node_key),
        },
        result_types=[result_types],
    )

    # Insert fused op at a position where all its inputs are available.
    # This must be AFTER all external input producers, not just before the
    # first subgraph op — the subgraph may span across non-fusible ops
    # (e.g. all_reduce in multi-GPU) that produce inputs consumed by later
    # subgraph ops.
    block = subgraph.ops[0].parent
    block_ops_order = {op: i for i, op in enumerate(block.ops)}
    first_sg_pos = block_ops_order[subgraph.ops[0]]

    # Find the latest input producer position in the block.
    latest_input_pos = -1
    for inp in subgraph.inputs:
        producer = inp.owner
        if producer in block_ops_order:
            latest_input_pos = max(latest_input_pos, block_ops_order[producer])

    if latest_input_pos >= first_sg_pos:
        # An external input is produced AFTER the first subgraph op.
        # Insert the fused op just after that input producer.
        anchor_op = list(block.ops)[latest_input_pos]
        block.insert_op_after(fused_op, anchor_op)
    else:
        # Normal case: all inputs are available before the subgraph.
        block.insert_op_before(fused_op, subgraph.ops[0])

    # Replace each subgraph output's uses with the corresponding fused op output
    for i, out_val in enumerate(subgraph.outputs):
        out_val.replace_by(fused_op.outputs[i])

    # Erase the original subgraph ops in reverse order.
    # safe_erase=False skips the use-check, which is needed because internal
    # operand references between subgraph ops may still exist at erase time.
    for op in reversed(subgraph.ops):
        block.erase_op(op, safe_erase=False)
