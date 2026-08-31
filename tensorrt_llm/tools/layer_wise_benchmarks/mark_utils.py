# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nvtx

from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate, Deepseekv3MoE
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate, DeepseekV4MoE
from tensorrt_llm._torch.models.modeling_kimi_linear import (
    KimiK3MoEGate,
    KimiK3MoERuntime,
    KimiMLARuntime,
)
from tensorrt_llm._torch.models.modeling_nemotron_h import MLPLayer, NemotronHMOE
from tensorrt_llm._torch.models.modeling_qwen3_next import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
from tensorrt_llm._torch.modules.mla import MLA
from tensorrt_llm._torch.moe.fused_moe.interface import MoE


def mark_ranges():
    DeepseekV3Gate.forward = nvtx.annotate("DeepseekV3Gate")(DeepseekV3Gate.forward)
    Deepseekv3MoE.forward = nvtx.annotate("Deepseekv3MoE")(Deepseekv3MoE.forward)
    MLPLayer.forward = nvtx.annotate("MLPLayer")(MLPLayer.forward)
    NemotronHMOE.forward = nvtx.annotate("NemotronHMOE")(NemotronHMOE.forward)
    Qwen3NextGatedDeltaNet.forward = nvtx.annotate("Qwen3NextGatedDeltaNet")(
        Qwen3NextGatedDeltaNet.forward
    )
    Qwen3NextSparseMoeBlock.forward = nvtx.annotate("Qwen3NextSparseMoeBlock")(
        Qwen3NextSparseMoeBlock.forward
    )
    # Kimi K3. KDA runs directly through `KimiKDALinearAttention`.
    # `KimiK3MLAAttention` overrides `MLA.forward`, so its range is on the
    # `KimiMLARuntime` wrapper. The gate is entered through `compute_logits`,
    # not `forward`. Its MLPs are the shared `GatedMLP`.
    KimiKDALinearAttention.forward = nvtx.annotate("KimiKDALinearAttention")(
        KimiKDALinearAttention.forward
    )
    KimiMLARuntime.forward = nvtx.annotate("KimiMLARuntime")(KimiMLARuntime.forward)
    KimiK3MoERuntime.forward = nvtx.annotate("KimiK3MoERuntime")(KimiK3MoERuntime.forward)
    KimiK3MoEGate.compute_logits = nvtx.annotate("KimiK3MoEGate")(KimiK3MoEGate.compute_logits)
    MLA.forward = nvtx.annotate("MLA")(MLA.forward)
    Attention.forward = nvtx.annotate("Attention")(Attention.forward)
    MoE.forward = nvtx.annotate("MoE")(MoE.forward)
    GatedMLP.forward = nvtx.annotate("GatedMLP")(GatedMLP.forward)
    Mamba2Mixer.forward = nvtx.annotate("Mamba2Mixer")(Mamba2Mixer.forward)

    # DeepseekV4Attention is covered by the MLA line above; these two are not.
    DeepseekV4Gate.forward = nvtx.annotate("DeepseekV4Gate")(DeepseekV4Gate.forward)
    DeepseekV4MoE.forward = nvtx.annotate("DeepseekV4MoE")(DeepseekV4MoE.forward)

    # mHC has no forward; DeepseekV4DecoderLayer calls these three directly.
    mHC.pre_mapping = nvtx.annotate("mHC pre_mapping")(mHC.pre_mapping)
    mHC.fused_hc = nvtx.annotate("mHC fused_hc")(mHC.fused_hc)
    mHC.post_mapping = nvtx.annotate("mHC post_mapping")(mHC.post_mapping)
