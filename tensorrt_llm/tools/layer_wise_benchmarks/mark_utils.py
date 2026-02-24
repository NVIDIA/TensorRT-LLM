import nvtx

from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate, Deepseekv3MoE
from tensorrt_llm._torch.models.modeling_nemotron_h import MLPLayer, NemotronHMOE
from tensorrt_llm._torch.models.modeling_qwen3_next import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from tensorrt_llm._torch.modules.attention import MLA, Attention
from tensorrt_llm._torch.modules.fused_moe.interface import MoE
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer


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
    MLA.forward = nvtx.annotate("MLA")(MLA.forward)
    Attention.forward = nvtx.annotate("Attention")(Attention.forward)
    MoE.forward = nvtx.annotate("MoE")(MoE.forward)
    GatedMLP.forward = nvtx.annotate("GatedMLP")(GatedMLP.forward)
    Mamba2Mixer.forward = nvtx.annotate("Mamba2Mixer")(Mamba2Mixer.forward)
