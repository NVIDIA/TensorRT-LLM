# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 family: XHToken Spark-X2.5/X2.6 (``Spark2_5ForCausalLM``).

Dense decoders with a 3:1 hybrid of sliding-window and full-attention layers,
per-layer-type RoPE (full layers rotate a quarter of the head), a headwise
sigmoid attention output gate, an erf-GELU gated MLP and tied embeddings.
One target covers the published BF16 checkpoints at tp1 on SM10x.
"""
