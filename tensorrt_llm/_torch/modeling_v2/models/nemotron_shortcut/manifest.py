# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local tensor windows shared by the text decoder weight loaders."""

_PREFIX = "model.language_model."
_MTP_PREFIX = _PREFIX + "mtp.layers.0."


def excluded_prefixes(mtp_enabled):
    prefixes = ("model.vision_tower.", "model.multi_modal_projector.")
    return prefixes if mtp_enabled else (*prefixes, _PREFIX + "mtp.")


def _fp32(tensor):
    return tensor.float()


def _squeeze_conv(tensor):
    return tensor.squeeze(1)


def row_window(start, width):
    return (slice(start, start + width),)


def col_window(start, width):
    return (slice(None), slice(start, start + width))


def block_columns(window, elements_per_block):
    """Map a row or column shard to the equivalent window in a packed matrix."""
    if not window or len(window) == 1:
        return window
    rows, cols = window
    if cols.step not in (None, 1) or any(
        bound is not None and bound % elements_per_block for bound in (cols.start, cols.stop)
    ):
        raise ValueError(f"Cannot shard {cols} on {elements_per_block}-element boundaries")
    start = None if cols.start is None else cols.start // elements_per_block
    stop = None if cols.stop is None else cols.stop // elements_per_block
    return rows, slice(start, stop)


def output_head_entry(core, head):
    local_vocab = head.shape[0]
    if local_vocab * core.tp_size != core.w["embed"].shape[0]:
        raise ValueError("The vocabulary must divide evenly across the output projection ranks.")
    return [
        (
            "lm_head.weight",
            row_window(core.slices["tp_rank"] * local_vocab, local_vocab),
            None,
            None,
        )
    ]


def check_checkpoint_keys(weights, expected, *, mtp_enabled):
    ignored = excluded_prefixes(mtp_enabled)
    provided = {key for key in weights if not key.startswith(ignored)}
    if expected != provided:
        missing, extra = sorted(expected - provided), sorted(provided - expected)
        raise ValueError(f"Checkpoint manifest mismatch: missing {missing[:8]}, extra {extra[:8]}")


def copy_checked(target, source, key):
    if source.shape != target.shape or source.dtype != target.dtype:
        raise ValueError(
            f"{key}: expected {target.shape} {target.dtype}, got {source.shape} {source.dtype}"
        )
    target.copy_(source, non_blocking=True)


def dynamic_placements(core):
    return (
        (prefix, placement)
        for prefix, placement in core.expert_placements.items()
        if placement.layer_load_balancer.is_dynamic_routing()
    )


def pad_expert_weight(weight, shape):
    """Keep the checkpoint mapping when the native layout already fits."""
    if tuple(weight.shape[-2:]) == tuple(shape):
        return weight
    rows, cols = weight.shape[-2:]
    padded = weight.new_zeros((*weight.shape[:-2], *shape))
    padded[..., :rows, :cols].copy_(weight)
    return padded


def expert_ids(selection):
    return range(selection.start, selection.stop) if isinstance(selection, slice) else selection


def load_unquantized(target, entries, weights):
    for key, window, rows, transform in entries:
        if window and isinstance(window[0], list):
            for slot, expert in enumerate(window[0]):
                copy_checked(target[slot], weights[key][(expert, *window[1:])], key)
            continue
        source = weights[key][window or (slice(None),)]
        if transform is not None:
            source = transform(source)
        copy_checked(target if rows is None else target[rows], source, key)


def build_manifest(core):
    """Map parameters to checkpoint keys, source windows, destination rows and transforms."""
    r = core.slices["tp_rank"]
    expert_rank = core.slices["moe_ep_rank"]
    moe_tp_rank = core.slices["moe_tp_rank"]
    tp = core.tp_size
    nh = core.gdp_heads
    hv_l, gk_l = core.gdp_v, core.gdp_k
    hd, di = core.head_dim, core.index_dim

    # Each rank takes the same head window from every Householder slot.
    conv_sections = [hv_l] * core.m + [gk_l] * (core.m + 1)
    projection_sections = [hv_l, *conv_sections, *([nh] * (core.m + 1))]

    def segment_windows(key, widths, transform=None):
        windows = []
        offset = 0
        for width in widths:
            source = row_window(offset * tp + r * width, width)
            windows.append((key, source, row_window(offset, width), transform))
            offset += width
        return windows

    def connection(i: int, conn: str, ckpt: str) -> dict:
        p = f"{_PREFIX}layers.{i}.{ckpt}."
        return {
            f"l{i}_{conn}_read": [(p + "read_logit", None, None, None)],
            f"l{i}_{conn}_write": [(p + "write_logit", None, None, None)],
            f"l{i}_{conn}_retain": [(p + "retention_logit", None, None, None)],
        }

    def attention(dst: str, a: str) -> dict:
        q_w, kv_w = core.heads_q * hd, core.heads_kv * hd
        # KV heads split like the query heads when there are at least tp of
        # them; otherwise adjacent ranks replicate a single KV head.
        kv_src = (r * core.heads_kv_global // tp) * hd
        return {
            f"{dst}qkv": [
                (a + "q_proj.weight", row_window(r * q_w, q_w), row_window(0, q_w), None),
                (a + "k_proj.weight", row_window(kv_src, kv_w), row_window(q_w, kv_w), None),
                (a + "v_proj.weight", row_window(kv_src, kv_w), row_window(q_w + kv_w, kv_w), None),
            ],
            f"{dst}o": [(a + "o_proj.weight", col_window(r * q_w, q_w), None, None)],
            # The indexer's one query and one key head, replicated like the KV head.
            **(
                {
                    f"{dst}idx": [
                        (a + "indexer.q_proj.weight", None, row_window(0, di), None),
                        (a + "indexer.k_proj.weight", None, row_window(di, di), None),
                    ]
                }
                if core.index_topk is not None
                else {}
            ),
        }

    def moe(dst: str, p: str) -> dict:
        """The post-mixer norm (under block prefix `p`) and the MoE itself."""
        e = p + "moe."
        el, il, si = core.local_experts, core.expert_inter, core.shared_inter
        placement = core.expert_placements.get(dst)
        experts = (
            placement.initial_local_expert_ids
            if placement is not None
            else slice(expert_rank * el, (expert_rank + 1) * el)
        )
        return {
            f"{dst}pre_moe_norm": [(p + "pre_moe_layernorm.weight", None, None, None)],
            f"{dst}router": [(e + "gate.weight", None, None, None)],
            f"{dst}router_bias": [(e + "gate.e_score_correction_bias", None, None, None)],
            f"{dst}fc1_latent": [(e + "fc1_latent_proj.weight", None, None, None)],
            f"{dst}fc2_norm": [(e + "fc2_norm.weight", None, None, None)],
            f"{dst}fc2_latent": [(e + "fc2_latent_proj.weight", None, None, None)],
            f"{dst}experts_up": [
                (
                    e + "experts.up_proj",
                    (
                        experts,
                        slice(moe_tp_rank * il, (moe_tp_rank + 1) * il),
                    ),
                    None,
                    None,
                )
            ],
            f"{dst}experts_down": [
                (
                    e + "experts.down_proj",
                    (
                        experts,
                        slice(None),
                        slice(moe_tp_rank * il, (moe_tp_rank + 1) * il),
                    ),
                    None,
                    None,
                )
            ],
            f"{dst}shared_up": [
                (e + "shared_experts.up_proj.weight", row_window(r * si, si), None, None)
            ],
            f"{dst}shared_down": [
                (e + "shared_experts.down_proj.weight", col_window(r * si, si), None, None)
            ],
        }

    out: dict = {}
    for i, kind in enumerate(core.kinds):
        p = f"{_PREFIX}layers.{i}."
        out[f"l{i}_norm_in"] = [(p + "input_layernorm.weight", None, None, None)]
        if kind in ("linear_attention", "mlp"):
            out.update(connection(i, "res", "residual_connection"))
        else:
            out.update(connection(i, "mix", "mixer_residual_connection"))
            out.update(connection(i, "moe", "moe_residual_connection"))

        if kind in ("linear_attention", "shortcut_moe_gdp"):
            g = p + "mixer."
            out[f"l{i}_gdp_in"] = segment_windows(g + "in_proj.weight", projection_sections)
            out[f"l{i}_gdp_conv"] = segment_windows(
                g + "conv1d.weight", conv_sections, _squeeze_conv
            )
            out[f"l{i}_gdp_a_log"] = [(g + "A_log", row_window(r * nh, nh), None, None)]
            out[f"l{i}_gdp_dt_bias"] = [(g + "dt_bias", row_window(r * nh, nh), None, _fp32)]
            out[f"l{i}_gdp_norm"] = [(g + "norm.weight", row_window(r * hv_l, hv_l), None, None)]
            out[f"l{i}_gdp_out"] = [(g + "out_proj.weight", col_window(r * hv_l, hv_l), None, None)]

        if kind == "shortcut_moe_attn":
            out.update(attention(f"l{i}_", p + "self_attn."))

        if kind == "mlp":
            out[f"l{i}_up"] = [
                (p + "mlp.up_proj.weight", row_window(r * core.inter, core.inter), None, None)
            ]
            out[f"l{i}_down"] = [
                (p + "mlp.down_proj.weight", col_window(r * core.inter, core.inter), None, None)
            ]

        if kind in ("shortcut_moe_attn", "shortcut_moe_gdp"):
            out[f"l{i}_shortcut_read"] = [(p + "shortcut_read.logit", None, None, None)]
            out[f"l{i}_shortcut_norm"] = [(p + "shortcut_input_layernorm.weight", None, None, None)]
            out.update(moe(f"l{i}_", p))

    if core.mtp_enabled:
        for name, key in (
            ("enorm", "enorm"),
            ("hnorm", "hnorm"),
            ("eh", "eh_proj"),
            ("norm_in", "block.input_layernorm"),
            ("final_norm", "final_layernorm"),
        ):
            out[f"mtp_{name}"] = [(_MTP_PREFIX + key + ".weight", None, None, None)]
        out.update(attention("mtp_", _MTP_PREFIX + "block.self_attn."))
        out.update(moe("mtp_", _MTP_PREFIX + "block."))
    out["readout"] = [(_PREFIX + "residual_readout.logit", None, None, None)]
    out["final_norm"] = [(_PREFIX + "norm.weight", None, None, None)]
    out["embed"] = [(_PREFIX + "embed_tokens.weight", None, None, None)]

    return out
