"""Builder ops for swapping attention backends on a meta model before load."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.attention import (
    Attention,
    AttentionCallable,
    AttentionFunction,
    MaskedAttentionCallable,
    MaskedAttentionFunction,
)


def set_attention_module_op(
    attention: AttentionFunction | AttentionCallable | None = None,
    masked_attention: MaskedAttentionFunction | MaskedAttentionCallable | None = None,
) -> ModuleOps:
    """Build a ``ModuleOps`` that overrides the attention callables on every
    ``Attention`` submodule of a model. Applied via ``create_meta_model`` so
    the meta model is mutated before unit weights are assigned. Matcher returns False
    for models with no ``Attention`` submodules, so the op is a no-op there.
    Either or both slots may be supplied; *None* leaves that slot untouched.
    """
    fn = attention.to_callable() if isinstance(attention, AttentionFunction) else attention
    masked_fn = (
        masked_attention.to_callable() if isinstance(masked_attention, MaskedAttentionFunction) else masked_attention
    )

    def matcher(model: torch.nn.Module) -> bool:
        return any(isinstance(m, Attention) for m in model.modules())

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        for module in model.modules():
            if isinstance(module, Attention):
                if fn is not None:
                    module.attention_function = fn
                if masked_fn is not None:
                    module.masked_attention_function = masked_fn
        return model

    return ModuleOps(name="set_attention_backend", matcher=matcher, mutator=mutator)
