import nvtx

from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate, Deepseekv3MoE
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate, DeepseekV4MoE
from tensorrt_llm._torch.models.modeling_nemotron_h import MLPLayer, NemotronHMOE
from tensorrt_llm._torch.models.modeling_qwen3_next import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.fused_moe.interface import MoE
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC
from tensorrt_llm._torch.modules.mla import MLA


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

    # DeepSeek-V4. DeepseekV4Attention subclasses MLA without overriding forward, so
    # it is already covered by the MLA line above. The gate and the MoE block are not:
    # DeepseekV4Gate is a separate class from DeepseekV3Gate, and DeepseekV4MoE does
    # not implement the MoE interface, so neither inherits an annotated forward.
    DeepseekV4Gate.forward = nvtx.annotate("DeepseekV4Gate")(DeepseekV4Gate.forward)
    DeepseekV4MoE.forward = nvtx.annotate("DeepseekV4MoE")(DeepseekV4MoE.forward)

    # mHC has no forward: DeepseekV4DecoderLayer drives it through three entry points
    # and deliberately fuses/defers them across the layer boundary (hc_attn.post_mapping
    # + hc_ffn.pre_mapping collapse into one fused_hc, and post_mapping may resolve in
    # the NEXT layer). There is therefore no single region to wrap -- annotating the
    # three call sites is what makes mHC kernels attributable at all. Without these the
    # only way to bucket them is by exclusion, which silently absorbs anything else the
    # module map does not know about.
    mHC.pre_mapping = nvtx.annotate("mHC pre_mapping")(mHC.pre_mapping)
    mHC.fused_hc = nvtx.annotate("mHC fused_hc")(mHC.fused_hc)
    mHC.post_mapping = nvtx.annotate("mHC post_mapping")(mHC.post_mapping)
