#!/usr/bin/env python3
"""Render a layer subgraph JSON (produced by the LLM) into DOT + PNG.

Usage:
    python render_layer.py <layer.json> [--output <path>]

The JSON schema is defined in the skill's SKILL.md.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Op-target → (friendly_name, fill_color)
# ---------------------------------------------------------------------------

_OP_STYLES: Dict[str, Tuple[str, str]] = {
    "flashinfer_mla_with_cache": ("MLA Attention", "lightcoral"),
    "flashinfer_mla_prepare_metadata": ("MLA Metadata", "lavender"),
    "flashinfer_rms_norm": ("RMSNorm", "lightyellow"),
    "flashinfer_fused_add_rms_norm": ("Fused Add+RMSNorm", "lightyellow"),
    "torch_linear_simple": ("Linear", "lightgreen"),
    "linear": ("Linear", "lightgreen"),
    "finegrained_fp8_linear": ("Linear (FP8)", "lightgreen"),
    "torch_attention_sdpa": ("SDPA Attention", "lightcoral"),
    "torch_attention": ("Attention", "lightcoral"),
    "triton_rope_on_interleaved_qk_inputs": ("RoPE", "lightsalmon"),
    "torch_rope_with_qk_interleaving": ("RoPE", "lightsalmon"),
    "flashinfer_rope": ("RoPE", "lightsalmon"),
    "fused_swiglu_mlp": ("Fused SwiGLU MLP", "wheat"),
    "fused_finegrained_fp8_swiglu_mlp": ("Fused SwiGLU MLP (FP8)", "wheat"),
    "trtllm_moe_fused": ("MoE Fused", "lightpink"),
    "quant_finegrained_fp8_moe_fused": ("MoE Fused (FP8)", "lightpink"),
    "noaux_tc_op": ("Top-K Routing", "lightpink"),
    "mamba_ssm_prepare_metadata": ("SSM Metadata", "lightskyblue"),
    "triton_cached_ssm": ("Cached SSM", "lightskyblue"),
    "triton_rmsnorm_gated": ("Gated RMSNorm", "lightyellow"),
    "cuda_cached_causal_conv1d_wrapper": ("Causal Conv1D", "lightcyan"),
    "triton_fused_gdn_gating": ("GDN Gating", "#b3e5fc"),
    "fla_cached_gated_delta_rule": ("Gated Delta Rule", "#81d4fa"),
    "trtllm_fused_allreduce_residual_rmsnorm": ("Fused AllReduce+RMSNorm", "lightyellow"),
    "trtllm_dist_all_reduce": ("AllReduce", "lavender"),
    "mlir_fused": ("Fused Kernel", "lightyellow"),
    "split_with_sizes": ("Split", "lavender"),
    "view": ("View", "white"),
    "reshape": ("Reshape", "white"),
    "add.Tensor": ("Add", "lightgray"),
    "add": ("Add", "lightgray"),
    "mul.Tensor": ("Mul", "white"),
    "mul": ("Mul", "white"),
    "sub": ("Sub", "white"),
    "floordiv": ("FloorDiv", "white"),
    "eq": ("Eq", "white"),
    "silu": ("SiLU", "wheat"),
    "cat": ("Concat", "lightyellow"),
    "t": ("Transpose", "white"),
    "to.dtype": ("To dtype", "white"),
    "to.device": ("To device", "white"),
    "to": ("To", "white"),
    "getitem": ("GetItem", "lavender"),
    "begin_aux_stream_passthrough": ("AuxStream Begin", "lavender"),
    "end_aux_stream_passthrough": ("AuxStream End", "lavender"),
    "dsv3_router_gemm_op": ("Router GEMM", "lightpink"),
    "symm_mem_all_gather": ("AllGather", "lavender"),
}


def _simplify_op(op: str) -> str:
    parts = op.split(".")
    for prefix in ("torch", "ops", "aten", "auto_deploy", "trtllm", "dist", "prims"):
        if parts and parts[0] == prefix:
            parts = parts[1:]
    if parts and parts[-1] == "default":
        parts = parts[:-1]
    return ".".join(parts) if parts else op


def _get_style(op: str) -> Tuple[str, str]:
    simplified = _simplify_op(op)
    if simplified in _OP_STYLES:
        return _OP_STYLES[simplified]
    for key, style in _OP_STYLES.items():
        if len(key) >= 4 and (key in simplified or key in op):
            return style
    for key, style in _OP_STYLES.items():
        if len(key) < 4 and (simplified == key or key in simplified.split("_")):
            return style
    return (simplified, "white")


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_shape(s: str) -> str:
    return s.replace("x", "×") if s else s


def _sanitize_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _short_weight(name: str) -> str:
    s = re.sub(r"^model_layers_\d+_", "", name)
    s = re.sub(r"^model_", "", s)
    s = re.sub(r"_(weight|bias)$", "", s)
    return re.sub(r"_+", "_", s).strip("_")


# ---------------------------------------------------------------------------
# Label building
# ---------------------------------------------------------------------------


def _make_label_without_kernels(node: dict) -> str:
    friendly, _ = _get_style(node["op"])
    node_id = node["id"]
    shape = node.get("shape", "")
    weights = node.get("weight_inputs", [])

    out_tag = f"O[{_fmt_shape(shape)}]" if shape and shape != "?" else ""

    if "Linear" in friendly:
        if weights:
            wname = _short_weight(weights[0]["name"])
            wshape = _fmt_shape(weights[0].get("shape", ""))
            parts = [f"{friendly} ({wname})", f"W[{wshape}]"]
            if out_tag:
                parts.append(out_tag)
            return "\\n".join(parts)
        return f"{friendly}\\n{node_id}"

    if "RMSNorm" in friendly or "Fused" in friendly:
        parts = [friendly, node_id]
        for w in weights:
            parts.append(f"{_short_weight(w['name'])} W[{_fmt_shape(w.get('shape', ''))}]")
        if out_tag:
            parts.append(out_tag)
        return "\\n".join(parts)

    if "MoE" in friendly:
        parts = [friendly, node_id]
        for w in weights:
            parts.append(f"{_short_weight(w['name'])} W[{_fmt_shape(w.get('shape', ''))}]")
        if out_tag:
            parts.append(out_tag)
        return "\\n".join(parts)

    if "Top-K" in friendly or "Router" in friendly:
        parts = [friendly, node_id]
        for w in weights:
            parts.append(f"{_short_weight(w['name'])} W[{_fmt_shape(w.get('shape', ''))}]")
        if out_tag:
            parts.append(out_tag)
        return "\\n".join(parts)

    if "Attention" in friendly or "MLA" in friendly:
        parts = [friendly, node_id]
        for w in weights:
            wn = _short_weight(w["name"])
            ws = _fmt_shape(w.get("shape", ""))
            if wn:
                parts.append(f"+ {wn} W[{ws}]")
        if out_tag:
            parts.append(out_tag)
        return "\\n".join(parts)

    if friendly == "GetItem":
        return f"{node_id}\\n{out_tag}" if out_tag else node_id

    if friendly in ("View", "Reshape"):
        return f"{friendly} ({node_id})\\n{out_tag}" if out_tag else f"{friendly}\\n{node_id}"

    if friendly == "Add":
        base = f"Add (Residual)\\n{node_id}"
        return f"{base}\\n{out_tag}" if out_tag else base

    parts = [friendly, node_id]
    if out_tag:
        parts.append(out_tag)
    return "\\n".join(parts)


# ---------------------------------------------------------------------------
# Cluster configuration
# ---------------------------------------------------------------------------

_CLUSTER_CFG = {
    "mla": ("Self-Attention (MLA)", "#f0f0f0"),
    "attention": ("Self-Attention", "#f0f0f0"),
    "gdn": ("Linear Attention (GDN)", "#e1f5fe"),
    "mamba": ("Mamba Mixer (SSM)", "#e8f4fd"),
}

_FFN_CFG = ("FFN (MoE + Shared Experts)", "#fff0f0")

_SUB_LABELS = {
    "q_branch": "Q Branch",
    "kv_branch": "KV Branch",
    "rope": "RoPE",
    "moe_router": "MoE Router",
    "moe_experts": "MoE Experts",
    "shared_experts": "Shared Experts",
    "gdn_conv1d": "Causal Conv1D",
    "gdn_gating": "GDN Gating",
    "gdn_delta_rule": "Gated Delta Rule",
    "mamba_metadata": "SSM Metadata",
    "mamba_ssm": "State Space Model",
    "mamba_gated_norm": "Gated RMSNorm",
}

# ---------------------------------------------------------------------------
# Aux stream detection
# ---------------------------------------------------------------------------


def _detect_aux_streams(nodes: Dict[str, dict], edges: list) -> Dict[str, Set[str]]:
    """Detect begin/end aux_stream_passthrough pairs and collect nodes between them."""
    begins: Dict[str, str] = {}
    ends: Dict[str, str] = {}
    for nid in nodes:
        if "begin_aux_stream" in nid:
            suffix = nid.rsplit("_", 1)[-1]
            begins[suffix] = nid
        elif "end_aux_stream" in nid:
            suffix = nid.rsplit("_", 1)[-1]
            ends[suffix] = nid

    fwd_map: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        if e["from"] in nodes and e["to"] in nodes:
            fwd_map[e["from"]].append(e["to"])

    aux_streams: Dict[str, Set[str]] = {}
    for suffix, begin_id in begins.items():
        if suffix not in ends:
            continue
        end_id = ends[suffix]
        stream_nodes: Set[str] = set()
        queue = [begin_id]
        visited = {begin_id}
        while queue:
            cur = queue.pop(0)
            stream_nodes.add(cur)
            if cur == end_id:
                continue
            for nxt in fwd_map.get(cur, []):
                if nxt not in visited and nxt in nodes:
                    visited.add(nxt)
                    queue.append(nxt)
        aux_streams[f"aux_{suffix}"] = stream_nodes

    return aux_streams


# ---------------------------------------------------------------------------
# DOT generation
# ---------------------------------------------------------------------------


def generate_dot(data: dict) -> str:
    layer_num = data["layer"]
    nodes = {n["id"]: n for n in data["nodes"]}
    edges = data.get("edges", [])
    ext_inputs = {e["id"]: e for e in data.get("external_inputs", [])}

    # Build unique DOT ids
    all_ids = set(nodes.keys()) | set(ext_inputs.keys())
    dot_ids: Dict[str, str] = {}
    used: Set[str] = set()
    for name in sorted(all_ids):
        sid = _sanitize_id(name)
        base = sid
        c = 2
        while sid in used:
            sid = f"{base}_{c}"
            c += 1
        used.add(sid)
        dot_ids[name] = sid

    # Group nodes
    groups: Dict[str, List[str]] = defaultdict(list)
    for n in data["nodes"]:
        groups[n.get("group", "other")].append(n["id"])

    # Sub-group lookup
    sub_groups: Dict[str, str] = {}
    for n in data["nodes"]:
        sg = n.get("sub_group", "")
        if sg:
            sub_groups[n["id"]] = sg

    # Propagate sub-groups to ungrouped neighbors within the same group
    node_group = {n["id"]: n.get("group", "other") for n in data["nodes"]}
    fwd: Dict[str, List[str]] = defaultdict(list)
    bwd: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        if e["from"] in nodes and e["to"] in nodes:
            fwd[e["from"]].append(e["to"])
            bwd[e["to"]].append(e["from"])

    changed = True
    while changed:
        changed = False
        for nid in nodes:
            if nid in sub_groups:
                continue
            grp = node_group.get(nid, "")
            neighbor_subs = []
            for nb in bwd.get(nid, []) + fwd.get(nid, []):
                if node_group.get(nb) == grp and nb in sub_groups:
                    neighbor_subs.append(sub_groups[nb])
            if neighbor_subs:
                sub_groups[nid] = Counter(neighbor_subs).most_common(1)[0][0]
                changed = True

    # Detect aux stream groups
    aux_streams = _detect_aux_streams(nodes, edges)
    node_to_aux: Dict[str, str] = {}
    for aux_id, aux_nodes in aux_streams.items():
        for nid in aux_nodes:
            node_to_aux[nid] = aux_id

    lines = [
        f"digraph Layer{layer_num} {{",
        "    rankdir=TB;",
        "    newrank=true;",
        '    node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        '    edge [fontname="Helvetica", fontsize=10];',
        "",
    ]

    rendered: Set[str] = set()

    def emit_node(nid: str, indent: str = "    "):
        nd = nodes[nid]
        did = dot_ids[nid]
        _, fill = _get_style(nd["op"])
        tks = nd.get("trace_kernels", [])
        if tks:
            # Use HTML label to color kernel annotations blue
            base_label = _make_label_without_kernels(nd)
            kern_lines = []
            for tk in tks:
                name = tk.get("kernel", "?")
                dur = tk.get("duration_us", 0)
                name_esc = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                kern_lines.append(f'<FONT COLOR="blue">⚡ {name_esc} ({dur:.1f}µs)</FONT>')
            kern_html = "<BR/>".join(kern_lines)
            base_esc = base_label.replace("\\n", "<BR/>")
            base_esc = base_esc.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_label = f"<{base_esc}<BR/>{kern_html}>"
            lines.append(f'{indent}{did} [label={html_label}, fillcolor="{fill}"];')
        else:
            label = _make_label_without_kernels(nd)
            lines.append(f'{indent}{did} [label="{label}", fillcolor="{fill}"];')

    def _emit_nodes_with_aux(node_list: List[str], indent: str):
        """Emit nodes, wrapping aux-stream members in a nested sub-cluster."""
        aux_groups_here: Dict[str, List[str]] = defaultdict(list)
        non_aux: List[str] = []
        for n in node_list:
            if n in rendered:
                continue
            aux_id = node_to_aux.get(n)
            if aux_id:
                aux_groups_here[aux_id].append(n)
            else:
                non_aux.append(n)

        for aux_id, aux_nodes in aux_groups_here.items():
            lines.append(f"{indent}subgraph cluster_{aux_id} {{")
            lines.append(f'{indent}    label="Aux Stream";')
            lines.append(f"{indent}    style=dotted;")
            lines.append(f'{indent}    bgcolor="#f0f8ff";')
            for n in aux_nodes:
                emit_node(n, indent + "    ")
                rendered.add(n)
            lines.append(f"{indent}}}")
            lines.append("")

        for n in non_aux:
            emit_node(n, indent)
            rendered.add(n)

    def emit_sub_cluster(parent_nodes: List[str], sub_key: str, indent: str):
        sc_nodes = [n for n in parent_nodes if sub_groups.get(n) == sub_key]
        if not sc_nodes:
            return
        label = _SUB_LABELS.get(sub_key, sub_key)
        lines.append(f"{indent}subgraph cluster_{sub_key} {{")
        lines.append(f'{indent}    label="{label}";')
        lines.append(f"{indent}    style=dashed;")
        _emit_nodes_with_aux(sc_nodes, indent + "    ")
        lines.append(f"{indent}}}")
        lines.append("")

    # Norm nodes split: input_layernorm → attention, post_attention → FFN
    norm_nodes = groups.get("norm", [])
    attn_norms = [n for n in norm_nodes if "input_layernorm" in n.lower()]
    ffn_norms = [n for n in norm_nodes if "post_attention" in n.lower()]
    other_norms = [n for n in norm_nodes if n not in attn_norms and n not in ffn_norms]

    # Attention / MLA cluster
    for grp in ("mla", "attention"):
        grp_nodes = groups.get(grp, [])
        if not grp_nodes:
            continue
        cfg = _CLUSTER_CFG.get(grp, (grp, "white"))
        lines.append("    subgraph cluster_self_attn {")
        lines.append(f'        label="{cfg[0]}";')
        lines.append("        style=rounded;")
        lines.append(f'        bgcolor="{cfg[1]}";')
        lines.append("")
        for n in attn_norms:
            emit_node(n, "        ")
            rendered.add(n)
        for sc in ("q_branch", "kv_branch", "rope"):
            emit_sub_cluster(grp_nodes, sc, "        ")
        remaining = [n for n in grp_nodes if n not in rendered]
        _emit_nodes_with_aux(remaining, "        ")
        lines.append("    }")
        lines.append("")

    # GDN cluster
    gdn_nodes = groups.get("gdn", [])
    if gdn_nodes:
        cfg = _CLUSTER_CFG["gdn"]
        lines.append("    subgraph cluster_gdn {")
        lines.append(f'        label="{cfg[0]}";')
        lines.append("        style=rounded;")
        lines.append(f'        bgcolor="{cfg[1]}";')
        lines.append("")
        for sc in ("gdn_conv1d", "gdn_gating", "gdn_delta_rule"):
            emit_sub_cluster(gdn_nodes, sc, "        ")
        remaining = [n for n in gdn_nodes if n not in rendered]
        _emit_nodes_with_aux(remaining, "        ")
        lines.append("    }")
        lines.append("")

    # Mamba cluster
    mamba_nodes = groups.get("mamba", [])
    if mamba_nodes:
        cfg = _CLUSTER_CFG["mamba"]
        lines.append("    subgraph cluster_mamba {")
        lines.append(f'        label="{cfg[0]}";')
        lines.append("        style=rounded;")
        lines.append(f'        bgcolor="{cfg[1]}";')
        lines.append("")
        for sc in ("mamba_metadata", "mamba_ssm", "mamba_gated_norm"):
            emit_sub_cluster(mamba_nodes, sc, "        ")
        remaining = [n for n in mamba_nodes if n not in rendered]
        _emit_nodes_with_aux(remaining, "        ")
        lines.append("    }")
        lines.append("")

    # Standalone norms between clusters
    for n in other_norms:
        if n not in rendered:
            emit_node(n, "    ")
            rendered.add(n)

    # FFN cluster (merged moe + mlp)
    ffn_nodes = groups.get("moe", []) + groups.get("mlp", [])
    if ffn_nodes:
        lines.append("    subgraph cluster_ffn {")
        lines.append(f'        label="{_FFN_CFG[0]}";')
        lines.append("        style=rounded;")
        lines.append(f'        bgcolor="{_FFN_CFG[1]}";')
        lines.append("")
        for n in ffn_norms:
            if n not in rendered:
                emit_node(n, "        ")
                rendered.add(n)
        for sc in ("moe_router", "moe_experts", "shared_experts"):
            emit_sub_cluster(ffn_nodes, sc, "        ")
        remaining = [n for n in ffn_nodes if n not in rendered]
        _emit_nodes_with_aux(remaining, "        ")
        lines.append("    }")
        lines.append("")

    # Other nodes
    remaining = [n for n in groups.get("other", []) if n not in rendered]
    _emit_nodes_with_aux(remaining, "    ")

    # External input nodes
    if ext_inputs:
        lines.append("    // External inputs")
        lines.append("    subgraph cluster_inputs {")
        lines.append('        label="Inputs";')
        lines.append("        style=dashed;")
        for eid, edata in sorted(ext_inputs.items()):
            did = dot_ids[eid]
            label_text = edata.get("label", eid)
            shape = edata.get("shape", "")
            if shape and shape != "?":
                label_text += f"\\nO[{_fmt_shape(shape)}]"
            lines.append(f'        {did} [label="{_escape(label_text)}", fillcolor=lightblue];')
        lines.append("    }")
        lines.append("")

    # Edges
    lines.append("    // Edges")
    seen: Set[Tuple[str, str]] = set()
    for e in edges:
        src = e["from"]
        dst = e["to"]
        src_did = dot_ids.get(src)
        dst_did = dot_ids.get(dst)
        if not src_did or not dst_did:
            continue
        key = (src_did, dst_did)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"    {src_did} -> {dst_did};")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_png(dot_path: str, png_path: str, dpi: int = 150) -> bool:
    try:
        result = subprocess.run(
            ["dot", "-Tpng", f"-Gdpi={dpi}", "-o", png_path, dot_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"Error rendering PNG: {result.stderr}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print(
            "Error: 'dot' command not found. Install graphviz: apt install graphviz",
            file=sys.stderr,
        )
        return False
    except subprocess.TimeoutExpired:
        print("Error: graphviz rendering timed out (>120s)", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Render a layer subgraph JSON to DOT + PNG.")
    parser.add_argument("json_file", help="Path to layer subgraph JSON")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path without extension (default: derived from json filename)",
    )
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(json_path.read_text())
    output_base = args.output or str(json_path.with_suffix(""))

    dot_source = generate_dot(data)
    dot_path = f"{output_base}.dot"
    Path(dot_path).write_text(dot_source)
    print(f"DOT written: {dot_path}")

    png_path = f"{output_base}.png"
    if render_png(dot_path, png_path):
        print(f"PNG written: {png_path}")
    else:
        print(
            f"PNG failed. Manual render: dot -Tpng -Gdpi=150 -o {png_path} {dot_path}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
