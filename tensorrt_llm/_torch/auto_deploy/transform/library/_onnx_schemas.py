# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


from onnx import defs
from onnxscript.values import Opset

_TRT_DOMAIN_NAME = "trt"
_AUTO_DEPLOY_DOMAIN_NAME = "auto_deploy"

# public opset objects, used in ONNX translation functions
trt_opset = Opset(_TRT_DOMAIN_NAME, 1)
auto_deploy_opset = Opset(_AUTO_DEPLOY_DOMAIN_NAME, 1)


# ONNX Custom Op Registration for RoPE
_torch_rope_with_explicit_cos_sin_schema = defs.OpSchema(
    name="rope_with_explicit_cos_sin",
    domain=_AUTO_DEPLOY_DOMAIN_NAME,
    since_version=1,
    doc="Rope with explicit cos and sin caches.",
    inputs=[
        defs.OpSchema.FormalParameter(
            name="q",
            description="Q tensor",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="k",
            description="K tensor",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="cos",
            description="Cos cache",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="sin",
            description="Sin cache",
            type_str="T",
        ),
    ],
    outputs=[
        defs.OpSchema.FormalParameter(
            name="output",
            description="Output tensor",
            type_str="T",
        )
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        defs.OpSchema.Attribute(
            name="unsqueeze_dim",
            type=defs.OpSchema.AttrType.INT,
            description="Unsqueeze dimension. Must be 1 or 2.",
            required=True,
        ),
    ],
)


# ONNX Custom Op Registration for AttentionPlugin
_attention_plugin_schema = defs.OpSchema(
    name="AttentionPlugin",
    domain=_TRT_DOMAIN_NAME,
    since_version=1,
    doc="Fused RoPE + Attention operation for efficient inference.",
    inputs=[
        defs.OpSchema.FormalParameter(
            name="qkv",
            description="Concatenated Q, K, V tensors in shape [batch, seq_len, qkv_hidden_size]",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="past_key_values",
            description="Concatenated past K and V cache in shape [batch, 2, num_kv_heads, past_len, head_size]",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="context_lengths",
            description="Context lengths for each sequence in shape [batch]",
            type_str="T1",
        ),
        defs.OpSchema.FormalParameter(
            name="rope_rotary_cos_sin",
            description="Concatenated cos and sin values for RoPE in shape [max_seq_len, head_dim]",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="kvcache_start_index",
            description="KV cache start index for each sequence in shape [batch]",
            type_str="T1",
        ),
    ],
    outputs=[
        defs.OpSchema.FormalParameter(
            name="output",
            description="Attention output in shape [batch, seq_len, hidden_size]",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="present_key_values",
            description="Updated K and V cache",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Input and output data type for floating point tensors.",
        ),
        (
            "T1",
            ["tensor(int32)", "tensor(int64)"],
            "Input data type for integer tensors.",
        ),
    ],
    attributes=[
        defs.OpSchema.Attribute(
            name="enable_tree_attention",
            type=defs.OpSchema.AttrType.INT,
            description="Whether to enable tree attention (0 or 1).",
            required=True,
        ),
        defs.OpSchema.Attribute(
            name="head_size",
            type=defs.OpSchema.AttrType.INT,
            description="Size of each attention head.",
            required=True,
        ),
        defs.OpSchema.Attribute(
            name="num_kv_heads",
            type=defs.OpSchema.AttrType.INT,
            description="Number of key-value heads.",
            required=True,
        ),
        defs.OpSchema.Attribute(
            name="num_q_heads",
            type=defs.OpSchema.AttrType.INT,
            description="Number of query heads.",
            required=True,
        ),
    ],
)


# ONNX Custom Op Registration for torch_attention
_torch_attention_schema = defs.OpSchema(
    name="torch_attention",
    domain=_AUTO_DEPLOY_DOMAIN_NAME,
    since_version=1,
    doc="SDPA attention (with optional GQA) that supports bnsd and bsnd memory layouts.",
    inputs=[
        defs.OpSchema.FormalParameter(
            name="query",
            description="Query tensor [batch, seq_len_q/num_heads, num_heads/seq_len_q, head_dim]",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="key",
            description="Key tensor [batch, seq_len_k/num_kv_heads, num_kv_heads/seq_len_k, head_dim]",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="value",
            description="Value tensor [batch, seq_len_k/num_kv_heads, num_kv_heads/seq_len_k, head_dim]",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="attn_mask",
            description="Optional attention mask in [batch, num_heads, seq_len_q, seq_len_k] layout",
            type_str="T",
        ),
        defs.OpSchema.FormalParameter(
            name="sinks",
            description="Optional sinks tensor",
            type_str="T",
        ),
    ],
    outputs=[
        defs.OpSchema.FormalParameter(
            name="output",
            description="Attention output in the same layout as inputs",
            type_str="T",
        )
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Input and output data type for floating point tensors.",
        ),
    ],
    attributes=[
        defs.OpSchema.Attribute(
            name="dropout_p",
            type=defs.OpSchema.AttrType.FLOAT,
            description="Dropout probability.",
            required=False,
        ),
        defs.OpSchema.Attribute(
            name="is_causal",
            type=defs.OpSchema.AttrType.INT,
            description="Whether to apply causal masking (0 or 1).",
            required=False,
        ),
        defs.OpSchema.Attribute(
            name="scale",
            type=defs.OpSchema.AttrType.FLOAT,
            description="Attention scale factor.",
            required=False,
        ),
        defs.OpSchema.Attribute(
            name="sliding_window",
            type=defs.OpSchema.AttrType.INT,
            description="Sliding window size for attention.",
            required=False,
        ),
        defs.OpSchema.Attribute(
            name="logit_cap",
            type=defs.OpSchema.AttrType.FLOAT,
            description="Logit capping value.",
            required=False,
        ),
        defs.OpSchema.Attribute(
            name="layout",
            type=defs.OpSchema.AttrType.STRING,
            description="Memory layout: 'bnsd' or 'bsnd'.",
            required=False,
        ),
    ],
)


def register_onnx_schemas():
    """Register ONNX custom ops."""
    defs.register_schema(_torch_rope_with_explicit_cos_sin_schema)
    defs.register_schema(_torch_attention_schema)
    defs.register_schema(_attention_plugin_schema)
