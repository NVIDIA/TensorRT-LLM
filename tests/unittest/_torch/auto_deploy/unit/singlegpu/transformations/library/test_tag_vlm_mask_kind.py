import torch
from torch.fx import Graph, GraphModule


def _make_gm_with_torch_attention():
    g = Graph()
    q = g.placeholder("q")
    k = g.placeholder("k")
    v = g.placeholder("v")
    attn = g.call_function(
        torch.ops.auto_deploy.torch_attention,
        args=(q, k, v),
        kwargs={"layout": "bsnd"},
    )
    g.output(attn)
    gm = GraphModule(torch.nn.Module(), g)
    return gm, attn


def test_tag_vlm_mask_kind_applies_for_vlm_mapping():
    from tensorrt_llm._torch.auto_deploy.transform.library.tag_vlm_mask_kind import TagVlmMaskKind

    gm, attn_node = _make_gm_with_torch_attention()
    gm.meta = {
        "ad_is_vlm": True,
        "ad_mask_kind_by_module": {"model.layers.0.self_attn": "full"},
    }
    attn_node.meta["nn_module_stack"] = {"model.layers.0.self_attn": torch.nn.Module}

    tr = TagVlmMaskKind.from_kwargs(stage="post_export")
    gm2, info = tr._apply(gm, cm=None, factory=None, shared_config=None)  # type: ignore[arg-type]

    assert gm2 is gm
    assert info.skipped is False
    assert attn_node.meta.get("mask_kind") == "full"


def test_tag_vlm_mask_kind_skips_for_non_vlm():
    from tensorrt_llm._torch.auto_deploy.transform.library.tag_vlm_mask_kind import TagVlmMaskKind

    gm, attn_node = _make_gm_with_torch_attention()
    gm.meta = {
        "ad_is_vlm": False,
        "ad_mask_kind_by_module": {"model.layers.0.self_attn": "full"},
    }
    attn_node.meta["nn_module_stack"] = {"model.layers.0.self_attn": torch.nn.Module}

    tr = TagVlmMaskKind.from_kwargs(stage="post_export")
    _, info = tr._apply(gm, cm=None, factory=None, shared_config=None)  # type: ignore[arg-type]

    assert info.skipped is True
    assert attn_node.meta.get("mask_kind") is None
