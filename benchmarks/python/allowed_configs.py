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
from dataclasses import asdict, dataclass
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class BuildConfig:
    num_layers: int
    num_heads: int
    hidden_size: int
    vocab_size: int
    hidden_act: Optional[str]
    n_positions: int
    max_batch_size: int
    max_input_len: Optional[int] = None
    num_kv_heads: Optional[int] = None
    max_output_len: Optional[int] = None
    max_beam_width: int = 1
    # TRT builder_optimization_level from 0 to 5
    builder_opt: Optional[int] = None
    inter_size: Optional[int] = None
    rotary_dim: Optional[int] = None
    type_vocab_size: Optional[int] = None
    pre_norm: Optional[bool] = None
    do_layer_norm_before: Optional[bool] = None
    enable_qk_half_accum: bool = False
    enable_context_fmha: bool = True
    enable_multi_block_mode: bool = False
    # The enum name of PositionEmbeddingType
    # None means using the model family's default value defined in the ctor
    position_embedding_type: str = None
    # Only when position embedding is RoPE, this value makes sense, make
    # default value to be None, not the others to prevent misuse
    rotary_pct: Optional[float] = None
    bias: bool = True
    quantization: Optional[str] = None
    # use_custom_all_reduce gives better performance with NVLink
    use_custom_all_reduce: bool = True
    moe_num_experts: int = 0
    moe_top_k: int = 0
    use_alibi: bool = None
    remove_input_padding: bool = None
    parallel_attention: bool = None
    new_decoder_architecture: bool = None


@dataclass
class EncDecBuildConfig:
    num_layers: int
    num_heads: int
    hidden_size: int
    vocab_size: int
    hidden_act: Optional[str]
    max_batch_size: int
    n_positions: int = 0
    num_decoder_layers: Optional[int] = None
    head_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_buckets: int = 0
    max_distance: int = 0
    has_embedding_scale: bool = False
    normalize_before: Optional[bool] = None
    max_encoder_input_len: Optional[int] = None
    max_decoder_input_len: Optional[int] = None
    max_output_len: Optional[int] = None
    builder_opt: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.head_size is not None
        assert self.ffn_hidden_size is not None


@dataclass
class ModelConfig:
    name: str
    family: str
    benchmark_type: Literal["gpt", "bert", "enc_dec"]
    build_config: BuildConfig


_allowed_configs = {
    "gpt_350m":
    ModelConfig(name="gpt_350m",
                family="gpt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=24,
                    num_heads=16,
                    hidden_size=1024,
                    vocab_size=51200,
                    hidden_act='gelu',
                    n_positions=1024,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "gpt_1.5b":
    ModelConfig(name="gpt_1.5b",
                family="gpt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=48,
                    num_heads=25,
                    hidden_size=1600,
                    vocab_size=51200,
                    hidden_act='gelu',
                    n_positions=1024,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "gpt_175b":
    ModelConfig(name="gpt_175b",
                family="gpt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=96,
                    num_heads=96,
                    hidden_size=12288,
                    vocab_size=51200,
                    hidden_act='gelu',
                    n_positions=2048,
                    max_batch_size=64,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "gpt_350m_moe":
    ModelConfig(name="gpt_350m_moe",
                family="gpt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=24,
                    num_heads=16,
                    hidden_size=1024,
                    vocab_size=51200,
                    hidden_act='gelu',
                    n_positions=1024,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    moe_num_experts=8,
                    moe_top_k=1,
                )),
    "gpt_350m_sq_per_tensor":
    ModelConfig(name="gpt_350m_sq_per_tensor",
                family="gpt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=24,
                    num_heads=16,
                    hidden_size=1024,
                    vocab_size=51200,
                    hidden_act='gelu',
                    n_positions=1024,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    quantization="int8_sq_per_tensor",
                )),
    "gpt_350m_sq_per_token_channel":
    ModelConfig(name="gpt_350m_sq_per_token_channel",
                family="gpt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=24,
                    num_heads=16,
                    hidden_size=1024,
                    vocab_size=51200,
                    hidden_act='gelu',
                    n_positions=1024,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    quantization="int8_sq_per_token_channel",
                )),
    "gpt_next_2b":
    ModelConfig(name="gpt_next_2b",
                family="gpt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=24,
                    num_heads=16,
                    hidden_size=2048,
                    vocab_size=256000,
                    hidden_act='swiglu',
                    n_positions=1024,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    position_embedding_type='rope_gpt_neox',
                    rotary_pct=0.5,
                    bias=False,
                )),
    "opt_350m":
    ModelConfig(name="opt_350m",
                family="opt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=24,
                    num_heads=16,
                    hidden_size=1024,
                    vocab_size=50272,
                    hidden_act='relu',
                    n_positions=2048,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    pre_norm=False,
                    do_layer_norm_before=False,
                )),
    "opt_2.7b":
    ModelConfig(name="opt_2.7b",
                family="opt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=32,
                    num_heads=32,
                    hidden_size=2560,
                    vocab_size=50272,
                    hidden_act='relu',
                    n_positions=2048,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    pre_norm=False,
                    do_layer_norm_before=True,
                )),
    "opt_6.7b":
    ModelConfig(name="opt_6.7b",
                family="opt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=32,
                    num_heads=32,
                    hidden_size=4096,
                    vocab_size=50272,
                    hidden_act='relu',
                    n_positions=2048,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    pre_norm=False,
                    do_layer_norm_before=True,
                )),
    "opt_66b":
    ModelConfig(name="opt_66b",
                family="opt",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=64,
                    num_heads=72,
                    hidden_size=9216,
                    vocab_size=50272,
                    hidden_act='relu',
                    n_positions=2048,
                    max_batch_size=64,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    pre_norm=True,
                    do_layer_norm_before=True,
                )),
    "llama_7b":
    ModelConfig(name="llama_7b",
                family="llama",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=32,
                    num_heads=32,
                    hidden_size=4096,
                    vocab_size=32000,
                    hidden_act='silu',
                    n_positions=4096,
                    inter_size=11008,
                    max_batch_size=128,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "llama_13b":
    ModelConfig(name="llama_13b",
                family="llama",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=40,
                    num_heads=40,
                    hidden_size=5120,
                    vocab_size=32000,
                    hidden_act='silu',
                    n_positions=4096,
                    inter_size=13824,
                    max_batch_size=128,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "llama_30b":
    ModelConfig(name="llama_30b",
                family="llama",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=60,
                    num_heads=52,
                    hidden_size=6656,
                    vocab_size=32000,
                    hidden_act='silu',
                    n_positions=2048,
                    inter_size=17920,
                    max_batch_size=64,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "llama_70b":
    ModelConfig(name="llama_70b",
                family="llama",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=80,
                    num_heads=64,
                    num_kv_heads=8,
                    hidden_size=8192,
                    vocab_size=32000,
                    hidden_act='silu',
                    n_positions=4096,
                    inter_size=28672,
                    max_batch_size=64,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "llama_70b_long_context":
    ModelConfig(name="llama_70b_long_context",
                family="llama",
                benchmark_type="gpt",
                build_config=BuildConfig(num_layers=80,
                                         num_heads=64,
                                         num_kv_heads=8,
                                         hidden_size=8192,
                                         vocab_size=32000,
                                         hidden_act='silu',
                                         n_positions=4096,
                                         inter_size=28672,
                                         max_batch_size=16,
                                         max_input_len=8000,
                                         max_output_len=200,
                                         builder_opt=None,
                                         enable_multi_block_mode=True)),
    "llama_70b_long_generation":
    ModelConfig(name="llama_70b_long_generation",
                family="llama",
                benchmark_type="gpt",
                build_config=BuildConfig(num_layers=80,
                                         num_heads=64,
                                         num_kv_heads=8,
                                         hidden_size=8192,
                                         vocab_size=32000,
                                         hidden_act='silu',
                                         n_positions=4096,
                                         inter_size=28672,
                                         max_batch_size=64,
                                         max_input_len=200,
                                         max_output_len=16384,
                                         builder_opt=None,
                                         enable_multi_block_mode=True)),
    "llama_70b_sq_per_tensor":
    ModelConfig(name="llama_70b_sq_per_tensor",
                family="llama",
                benchmark_type="gpt",
                build_config=BuildConfig(num_layers=80,
                                         num_heads=64,
                                         num_kv_heads=8,
                                         hidden_size=8192,
                                         vocab_size=32000,
                                         hidden_act='silu',
                                         n_positions=4096,
                                         inter_size=28672,
                                         max_batch_size=128,
                                         max_input_len=512,
                                         max_output_len=200,
                                         builder_opt=None,
                                         quantization="int8_sq_per_tensor")),
    "mixtral_8x7b":
    ModelConfig(name="mixtral_8x7b",
                family="llama",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=32,
                    num_heads=32,
                    hidden_size=4096,
                    vocab_size=32000,
                    hidden_act='swiglu',
                    n_positions=2048,
                    inter_size=14336,
                    max_batch_size=128,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    moe_num_experts=8,
                    moe_top_k=2,
                )),
    "gptj_6b":
    ModelConfig(name="gptj_6b",
                family="gptj",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=28,
                    num_heads=16,
                    hidden_size=4096,
                    vocab_size=50401,
                    hidden_act='gelu',
                    n_positions=1024,
                    rotary_dim=64,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "gptneox_20b":
    ModelConfig(name="gptneox_20b",
                family="gptneox",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=44,
                    num_heads=64,
                    hidden_size=6144,
                    vocab_size=50432,
                    hidden_act='gelu',
                    n_positions=2048,
                    rotary_dim=24,
                    max_batch_size=16,
                    max_input_len=512,
                    max_output_len=512,
                    builder_opt=None,
                )),
    "chatglm_6b":
    ModelConfig(name="chatglm_6b",
                family="chatglm",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=28,
                    num_heads=32,
                    hidden_size=4096,
                    vocab_size=130528,
                    hidden_act='gelu',
                    n_positions=2048,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    remove_input_padding=False,
                )),
    "chatglm2_6b":
    ModelConfig(name="chatglm2_6b",
                family="chatglm2",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=28,
                    num_heads=32,
                    num_kv_heads=2,
                    hidden_size=4096,
                    vocab_size=65024,
                    hidden_act='swiglu',
                    n_positions=2048,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    remove_input_padding=False,
                )),
    "chatglm3_6b":
    ModelConfig(name="chatglm3_6b",
                family="chatglm3",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=28,
                    num_heads=32,
                    num_kv_heads=2,
                    hidden_size=4096,
                    vocab_size=65024,
                    hidden_act='swiglu',
                    n_positions=2048,
                    max_batch_size=256,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    remove_input_padding=False,
                )),
    "bloom_560m":
    ModelConfig(name="bloom_560m",
                family="bloom",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=24,
                    num_heads=16,
                    hidden_size=1024,
                    vocab_size=250880,
                    hidden_act=None,
                    n_positions=2048,
                    max_batch_size=8,
                    max_input_len=1024,
                    max_output_len=1024,
                    builder_opt=None,
                )),
    "bloom_176b":
    ModelConfig(name="bloom_176b",
                family="bloom",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=70,
                    num_heads=112,
                    hidden_size=14336,
                    vocab_size=250880,
                    hidden_act=None,
                    n_positions=2048,
                    max_batch_size=8,
                    max_input_len=1024,
                    max_output_len=1024,
                    builder_opt=None,
                )),
    "bert_base":
    ModelConfig(name="bert_base",
                family="bert",
                benchmark_type="bert",
                build_config=BuildConfig(
                    num_layers=12,
                    num_heads=12,
                    hidden_size=768,
                    vocab_size=30522,
                    type_vocab_size=2,
                    hidden_act='gelu',
                    n_positions=1024,
                    max_batch_size=256,
                    max_input_len=512,
                    builder_opt=None,
                    enable_qk_half_accum=False,
                    enable_context_fmha=False,
                )),
    "bert_large":
    ModelConfig(name="bert_large",
                family="bert",
                benchmark_type="bert",
                build_config=BuildConfig(
                    num_layers=24,
                    num_heads=16,
                    hidden_size=1024,
                    vocab_size=30522,
                    type_vocab_size=2,
                    hidden_act='gelu',
                    n_positions=1024,
                    max_batch_size=64,
                    max_input_len=512,
                    builder_opt=None,
                    enable_qk_half_accum=False,
                    enable_context_fmha=False,
                )),
    "falcon_rw_1b":
    ModelConfig(name="falcon_rw_1b",
                family="falcon",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=24,
                    num_heads=32,
                    hidden_size=2048,
                    vocab_size=50304,
                    hidden_act='gelu',
                    n_positions=2048,
                    max_batch_size=256,
                    max_input_len=1024,
                    max_output_len=1024,
                    builder_opt=None,
                    bias=True,
                    use_alibi=True,
                    parallel_attention=False,
                    new_decoder_architecture=False,
                )),
    "falcon_7b":
    ModelConfig(name="falcon_7b",
                family="falcon",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=32,
                    num_heads=71,
                    num_kv_heads=1,
                    hidden_size=4544,
                    vocab_size=65024,
                    hidden_act='gelu',
                    n_positions=2048,
                    max_batch_size=128,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    bias=False,
                    use_alibi=False,
                    parallel_attention=True,
                    new_decoder_architecture=False,
                )),
    "falcon_40b":
    ModelConfig(name="falcon_40b",
                family="falcon",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=60,
                    num_heads=128,
                    num_kv_heads=8,
                    hidden_size=8192,
                    vocab_size=65024,
                    hidden_act='gelu',
                    n_positions=2048,
                    max_batch_size=64,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    bias=False,
                    use_alibi=False,
                    parallel_attention=True,
                    new_decoder_architecture=True,
                )),
    "falcon_180b":
    ModelConfig(name="falcon_180b",
                family="falcon",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=80,
                    num_heads=232,
                    num_kv_heads=8,
                    hidden_size=14848,
                    vocab_size=65024,
                    hidden_act='gelu',
                    n_positions=2048,
                    max_batch_size=8,
                    max_input_len=1024,
                    max_output_len=1024,
                    builder_opt=None,
                    bias=False,
                    use_alibi=False,
                    parallel_attention=True,
                    new_decoder_architecture=True,
                )),
    "t5_small":
    ModelConfig(name="t5_small",
                family="t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=6,
                    num_heads=8,
                    head_size=64,
                    ffn_hidden_size=2048,
                    hidden_size=512,
                    vocab_size=32128,
                    hidden_act="relu",
                    n_positions=512,
                    num_buckets=32,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "t5_base":
    ModelConfig(name="t5_base",
                family="t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=12,
                    num_heads=12,
                    head_size=64,
                    ffn_hidden_size=3072,
                    hidden_size=768,
                    vocab_size=32128,
                    hidden_act="relu",
                    n_positions=512,
                    num_buckets=32,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "t5_large":
    ModelConfig(name="t5_large",
                family="t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=24,
                    num_heads=16,
                    head_size=64,
                    ffn_hidden_size=4096,
                    hidden_size=1024,
                    vocab_size=32128,
                    hidden_act="relu",
                    n_positions=512,
                    num_buckets=32,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "t5_3b":
    ModelConfig(name="t5_3b",
                family="t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=24,
                    num_heads=32,
                    head_size=128,
                    ffn_hidden_size=16384,
                    hidden_size=1024,
                    vocab_size=32128,
                    hidden_act="relu",
                    n_positions=512,
                    num_buckets=32,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "t5_11b":
    ModelConfig(name="t5_11b",
                family="t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=24,
                    num_heads=128,
                    head_size=128,
                    ffn_hidden_size=65536,
                    hidden_size=1024,
                    vocab_size=32128,
                    hidden_act="relu",
                    n_positions=512,
                    num_buckets=32,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "flan_t5_small":
    ModelConfig(name="flan_t5_small",
                family="flan_t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=8,
                    num_decoder_layers=8,
                    num_heads=6,
                    head_size=64,
                    ffn_hidden_size=1024,
                    hidden_size=512,
                    vocab_size=32128,
                    hidden_act="gelu_new",
                    n_positions=512,
                    num_buckets=32,
                    max_distance=128,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "flan_t5_base":
    ModelConfig(name="flan_t5_base",
                family="flan_t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=12,
                    num_decoder_layers=12,
                    num_heads=12,
                    head_size=64,
                    ffn_hidden_size=2048,
                    hidden_size=768,
                    vocab_size=32128,
                    hidden_act="gelu_new",
                    n_positions=512,
                    num_buckets=32,
                    max_distance=128,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "flan_t5_large":
    ModelConfig(name="flan_t5_large",
                family="flan_t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=24,
                    num_decoder_layers=24,
                    num_heads=16,
                    head_size=64,
                    ffn_hidden_size=2816,
                    hidden_size=1024,
                    vocab_size=32128,
                    hidden_act="gelu_new",
                    n_positions=512,
                    num_buckets=32,
                    max_distance=128,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "flan_t5_xl":
    ModelConfig(name="flan_t5_xl",
                family="flan_t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=24,
                    num_decoder_layers=24,
                    num_heads=32,
                    head_size=64,
                    ffn_hidden_size=5120,
                    hidden_size=2048,
                    vocab_size=32128,
                    hidden_act="gelu_new",
                    n_positions=512,
                    num_buckets=32,
                    max_distance=128,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "flan_t5_xxl":
    ModelConfig(name="flan_t5_xxl",
                family="flan_t5",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=24,
                    num_decoder_layers=24,
                    num_heads=64,
                    head_size=64,
                    ffn_hidden_size=10240,
                    hidden_size=4096,
                    vocab_size=32128,
                    hidden_act="gelu_new",
                    n_positions=0,
                    num_buckets=32,
                    max_distance=128,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "bart_large_cnn":
    ModelConfig(name="bart_large_cnn",
                family="bart",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=12,
                    num_decoder_layers=12,
                    num_heads=16,
                    head_size=64,
                    ffn_hidden_size=4096,
                    hidden_size=1024,
                    vocab_size=50265,
                    hidden_act="gelu",
                    n_positions=1024,
                    num_buckets=32,
                    has_embedding_scale=False,
                    normalize_before=False,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "mbart_large_50_many_to_one_mmt":
    ModelConfig(name="mbart_large_50_many_to_one_mmt",
                family="bart",
                benchmark_type="enc_dec",
                build_config=EncDecBuildConfig(
                    num_layers=12,
                    num_decoder_layers=12,
                    num_heads=16,
                    head_size=64,
                    ffn_hidden_size=4096,
                    hidden_size=1024,
                    vocab_size=250054,
                    hidden_act="relu",
                    n_positions=1024,
                    has_embedding_scale=True,
                    normalize_before=True,
                    max_batch_size=8,
                    max_encoder_input_len=1024,
                    max_decoder_input_len=1,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "baichuan_7b":
    ModelConfig(name="baichuan_7b",
                family="baichuan_7b",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=32,
                    num_heads=32,
                    hidden_size=4096,
                    vocab_size=64000,
                    hidden_act='silu',
                    n_positions=4096,
                    inter_size=11008,
                    max_batch_size=128,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "baichuan2_7b_chat":
    ModelConfig(name="baichuan2_7b_chat",
                family="baichuan_7b",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=32,
                    num_heads=32,
                    hidden_size=4096,
                    vocab_size=125696,
                    hidden_act='silu',
                    n_positions=4096,
                    inter_size=11008,
                    max_batch_size=128,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "baichuan_13b_chat":
    ModelConfig(name="baichuan_13b_chat",
                family="baichuan_13b",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=40,
                    num_heads=40,
                    hidden_size=5120,
                    vocab_size=64000,
                    hidden_act='silu',
                    n_positions=4096,
                    inter_size=13696,
                    max_batch_size=64,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "baichuan2_13b_chat":
    ModelConfig(name="baichuan2_13b_chat",
                family="baichuan_13b",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=40,
                    num_heads=40,
                    hidden_size=5120,
                    vocab_size=125696,
                    hidden_act='silu',
                    n_positions=4096,
                    inter_size=13696,
                    max_batch_size=64,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
    "internlm_chat_7b":
    ModelConfig(name="internlm_chat_7b",
                family="internlm",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=32,
                    num_heads=32,
                    num_kv_heads=32,
                    hidden_size=4096,
                    vocab_size=103168,
                    hidden_act='silu',
                    n_positions=2048,
                    inter_size=11008,
                    max_batch_size=128,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    bias=True,
                )),
    "internlm_chat_20b":
    ModelConfig(name="internlm_chat_20b",
                family="internlm",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=60,
                    num_heads=40,
                    num_kv_heads=40,
                    hidden_size=5120,
                    vocab_size=103168,
                    hidden_act='silu',
                    n_positions=4096,
                    inter_size=13824,
                    max_batch_size=64,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                    bias=False,
                )),
    "qwen_7b_chat":
    ModelConfig(name="qwen_7b_chat",
                family="qwen",
                benchmark_type="gpt",
                build_config=BuildConfig(num_layers=32,
                                         num_heads=32,
                                         hidden_size=4096,
                                         vocab_size=151936,
                                         hidden_act='silu',
                                         n_positions=8192,
                                         inter_size=22016,
                                         max_batch_size=128,
                                         max_input_len=512,
                                         max_output_len=200,
                                         builder_opt=None,
                                         bias=False)),
    "qwen_14b_chat":
    ModelConfig(name="qwen_14b_chat",
                family="qwen",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=40,
                    num_heads=40,
                    hidden_size=5120,
                    vocab_size=152064,
                    hidden_act='silu',
                    n_positions=8192,
                    inter_size=27392,
                    max_batch_size=64,
                    max_input_len=512,
                    max_output_len=200,
                    builder_opt=None,
                )),
}


def get_allowed_models(benchmark_type=None):
    if benchmark_type is None:
        return set(_allowed_configs.keys())
    else:
        return set(i.name for i in _allowed_configs.values()
                   if i.benchmark_type == benchmark_type)


def get_build_config(model_name):
    if model_name in _allowed_configs:
        return asdict(_allowed_configs[model_name].build_config)
    else:
        raise KeyError(f'Unexpected model: {model_name}. Please add the model '
                       'to allowed_configs.py')


def get_model_family(model_name):
    if model_name in _allowed_configs:
        return _allowed_configs[model_name].family
    else:
        raise KeyError(f'Unexpected model: {model_name}. Please add the model '
                       'to allowed_configs.py')


def get_benchmark_type(model_name):
    if model_name in _allowed_configs:
        return _allowed_configs[model_name].benchmark_type
    else:
        raise KeyError(f'Unexpected model: {model_name}. Please add the model '
                       'to allowed_configs.py')
