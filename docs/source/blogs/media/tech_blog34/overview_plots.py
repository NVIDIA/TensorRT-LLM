# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render decoder structure and measured WinoVAE workload profiles."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parent
INK, GREY, GREEN, BLUE = "#203044", "#697586", "#507C0C", "#2563A6"
COLORS = {"diffusers": "#9AA7B3", "native": BLUE, "winovae": GREEN}
NAMES = {"diffusers": "Diffusers BF16", "native": "BF16 Wan VAE", "winovae": "WinoVAE"}
WORKLOADS = [
    ("t2i_1024", "Image: 1024 × 1024"),
    ("t2v_720p_189f", "Video: 189 frames, 1280 × 720"),
    ("edge_480p_121f", "Video: 121 frames, 832 × 480"),
    ("bimanual", "Streaming: 832 × 480"),
]


# Intrinsic widths in CSS pixels. The SVG viewBox sets the aspect ratio.
DISPLAY_WIDTHS = {
    "decoder_graph": 480,
    "fusion": 480,
    "upsample_fold": 505,
    "causal_first_frame": 450,
    "transform_stages": 425,
    "channel_flow": 480,
    "spatial_tiles": 465,
    "coordinate_gemm": 820,
    "cost_flops": 520,
    "cost_workspace": 520,
    "cudnn_sweep": 545,
    "latency": 465,
    "halo_latency": 505,
    "halo_exchange": 495,
    "strip_allgather": 540,
    "nvfp4_error": 490,
}


def article_svg(svg: str, name: str) -> str:
    if name in DISPLAY_WIDTHS:
        svg = re.sub(
            r'(<svg\b[^>]*?) width="[^"]+" height="[^"]+"',
            rf'\g<1> width="{DISPLAY_WIDTHS[name]}"',
            svg,
            count=1,
        )
    return "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"


def save(fig: plt.Figure, name: str) -> None:
    rights = "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    fig.savefig(
        ROOT / f"{name}.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.16,
        facecolor="white",
        metadata={"Copyright": rights},
    )
    target = ROOT / f"{name}.svg"
    fig.savefig(
        target, bbox_inches="tight", pad_inches=0.16, facecolor="white", metadata={"Rights": rights}
    )
    target.write_text(article_svg(target.read_text(), name))
    plt.close(fig)


def node(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: str = "#EAF0F6",
    size: float = 13,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor="#B3BFCA",
            linewidth=0.8,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax: plt.Axes, start: tuple, end: tuple) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.5})


def decoder_graph() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 8.8))
    ax.set(xlim=(0, 6.4), ylim=(0, 8.8))
    ax.axis("off")
    rows = [
        ("Input", "Latents", "48 ch · 1 frame\n45 × 80", 0.45),
        (
            "conv_in\n+ mid",
            "conv_in\nResBlock\nAttention\nResBlock",
            "1024 ch · 1 frame\n45 × 80",
            1.12,
        ),
        (
            "up0",
            "3 ResBlocks\nTime conv 3×1×1\nUpsample 2×\nSpatial conv 3×3",
            "1024 ch · 2 frames\n90 × 160",
            1.12,
        ),
        (
            "up1",
            "3 ResBlocks\nTime conv 3×1×1\nUpsample 2×\nSpatial conv 3×3",
            "1024 ch · 4 frames\n180 × 320",
            1.12,
        ),
        ("up2", "3 ResBlocks\nUpsample 2×\nSpatial conv 3×3", "512 ch · 4 frames\n360 × 640", 0.86),
        ("up3", "3 ResBlocks", "256 ch · 4 frames\n360 × 640", 0.45),
        ("Output", "RMSNorm\nSiLU\nconv_out\nUnpatchify", "3 ch · 4 frames\n720 × 1280", 1.12),
    ]
    y = 8.45
    for i, (name, label, dims, height) in enumerate(rows):
        y -= height
        ax.text(0.58, y + height / 2, name, ha="center", va="center", fontsize=12, weight="bold")
        node(ax, 1.3, y, 3.0, height, label, "#E3EFD3" if name in ("up2", "up3") else "#EAF0F6", 13)
        ax.text(4.55, y + height / 2, dims, va="center", fontsize=12, color=INK)
        if i != len(rows) - 1:
            arrow(ax, (2.8, y - 0.04), (2.8, y - 0.26))
        y -= 0.30
    save(fig, "decoder_graph")


def causal_first_frame() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.set(xlim=(0, 6.4), ylim=(0, 5.8))
    ax.axis("off")
    for first, top in [(False, 5.65), (True, 2.80)]:
        ax.text(
            0.1,
            top,
            "First frame: zero history" if first else "With temporal history",
            fontsize=15,
            weight="bold",
            color=INK,
            va="top",
        )
        tile_y = top - 1.65
        line_y = tile_y - 0.60
        for k, x in enumerate([0.15, 1.62, 3.09]):
            zero = first and k < 2
            label = (
                [r"$x_{-2}=0$", r"$x_{-1}=0$", r"$x_0$"][k]
                if first
                else [r"$x_{t-2}$", r"$x_{t-1}$", r"$x_t$"][k]
            )
            ax.text(
                x + 0.45,
                tile_y + 1.08,
                label,
                ha="center",
                fontsize=14,
                color=GREY if zero else INK,
            )
            ax.add_patch(
                Rectangle(
                    (x, tile_y),
                    0.9,
                    0.9,
                    facecolor="#F0F2F4" if zero else "#EAF0F6",
                    edgecolor="#9AA7B3",
                    linewidth=0.8,
                )
            )
            if zero:
                ax.text(
                    x + 0.45,
                    tile_y + 0.45,
                    "0",
                    ha="center",
                    va="center",
                    fontsize=24,
                    color="#9AA7B3",
                )
            else:
                for n in range(1, 5):
                    ax.plot([x + n * 0.18] * 2, [tile_y, tile_y + 0.9], color="white", lw=0.7)
                    ax.plot([x, x + 0.9], [tile_y + n * 0.18] * 2, color="white", lw=0.7)
                ax.add_patch(
                    Rectangle(
                        (x + 0.18, tile_y + 0.18),
                        0.54,
                        0.54,
                        facecolor="#B5DC6A",
                        edgecolor=GREEN,
                        alpha=0.8,
                    )
                )
                ax.annotate(
                    "",
                    xy=(x + 0.45, line_y),
                    xytext=(x + 0.45, tile_y - 0.02),
                    arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.4},
                )
                ax.text(x + 0.56, tile_y - 0.28, rf"$W_{k}\ast$", fontsize=12, color=GREEN)
        start_x = 3.54 if first else 0.6
        ax.plot([start_x, 5.65], [line_y, line_y], color=GREEN, lw=1.4)
        ax.annotate(
            "",
            xy=(5.65, tile_y + 0.1),
            xytext=(5.65, line_y),
            arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.4},
        )
        ax.text(
            4.7,
            line_y,
            "+ b" if first else "Σ + b",
            ha="center",
            va="center",
            fontsize=13,
            color=GREEN,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
        )
        node(ax, 5.2, tile_y + 0.1, 0.9, 0.7, r"$y_0$" if first else r"$y_t$", "#E3EFD3", 17)
    save(fig, "causal_first_frame")


def transform_stages() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.set(xlim=(0, 6.4), ylim=(-0.12, 3.1))
    ax.axis("off")
    for x, label, equation, color in [
        (0.12, "Input tile", r"$v=B^{T}d$", "#DCE9F7"),
        (3.81, "Filter", r"$u=Gg$", "#FBE5D3"),
    ]:
        ax.text(x + 1.22, 2.84, label, ha="center", va="center", fontsize=14)
        node(ax, x, 2.05, 2.44, 0.55, equation, color, 15)
    arrow(ax, (1.34, 2.00), (2.60, 1.54))
    arrow(ax, (5.03, 2.00), (3.80, 1.54))
    node(ax, 2.0, 0.95, 2.4, 0.5, r"$m_i=v_i u_i$", "#E3EFD3", 15)
    arrow(ax, (3.2, 0.90), (3.2, 0.54))
    node(ax, 2.0, 0.02, 2.4, 0.45, r"$y=A^{T}m$", "#EAF0F6", 15)
    save(fig, "transform_stages")


def decoder_breakdown(data: dict) -> None:
    vals = list(data["motivation"]["decoder_breakdown"]["seconds"].values())
    labels = [
        "Convolutions (cuDNN)",
        "Channel RMSNorm",
        "SiLU, bias, residual, copies",
        "Nearest upsample",
        "Upsample shortcut",
        "Other",
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.1), layout="constrained")
    ax.barh(labels, vals, color=["#76B900"] + ["#B5DC6A"] * 5, height=0.64)
    for i, v in enumerate(vals):
        ax.text(v + 0.08, i, f"{v:.1f} s ({100 * v / sum(vals):.0f}%)", va="center", fontsize=11)
    ax.invert_yaxis()
    ax.set(xlim=(0, 6.3), xlabel="Decoder time (seconds)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.xaxis.label.set_size(11)
    save(fig, "decoder_breakdown")


def hardware(data: dict, key: str) -> None:
    rows = data[key]["results"]
    fig = plt.figure(figsize=(6.4, 7.1), layout="constrained")
    panels = fig.subfigures(4, 1)
    for i, (shape, title) in enumerate(WORKLOADS):
        panels[i].suptitle(title, x=0.02, ha="left", fontsize=12, weight="bold")
        axes = panels[i].subplots(1, 2)
        for ax, op in zip(axes, ["encode", "decode"]):
            target = shape
            metric = "median_ms"
            subtitle = op.capitalize()
            if shape == "bimanual":
                target = "bimanual_cond_480x832" if op == "encode" else "bimanual_480x832_901f"
                if op == "decode":
                    metric = "step_ms"
                    subtitle += ": 2 latents → 8 frames"
                else:
                    subtitle += ": 1 frame"
            selected = {
                r["backend"]: r[metric] for r in rows if r["shape"] == target and r["op"] == op
            }
            vals = [selected[b] for b in NAMES]
            ax.bar(np.arange(3), vals, color=[COLORS[b] for b in NAMES], width=0.65)
            for x, v in enumerate(vals):
                ax.text(
                    x,
                    v + max(vals) * 0.03,
                    f"{v:.0f}" if v >= 100 else f"{v:.1f}",
                    ha="center",
                    fontsize=11,
                )
            ax.set_ylim(0, max(vals) * 1.32)
            ax.set_xticks([])
            ax.set_title(subtitle + " (ms)", fontsize=10, loc="left", pad=3)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
    fig.legend(
        [Patch(color=COLORS[b]) for b in NAMES],
        list(NAMES.values()),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    save(fig, key)


def scaling(data: dict) -> None:
    rows = data["b200_strong_scaling"]["results"]
    for op in ["encode", "decode"]:
        fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.2), layout="constrained")
        for ax, (shape, title) in zip(axs.flat, WORKLOADS):
            target = shape
            metric = "median_ms"
            if shape == "bimanual":
                target = "bimanual_cond_480x832" if op == "encode" else "bimanual_480x832_901f"
                title = "Streaming: " + (
                    "1 input frame" if op == "encode" else "2 latents → 8 frames"
                )
                if op == "decode":
                    metric = "step_ms"
            for backend, marker in zip(NAMES, ["^", "s", "o"]):
                selected = sorted(
                    (
                        r
                        for r in rows
                        if r["backend"] == backend and r["shape"] == target and r["op"] == op
                    ),
                    key=lambda r: r["gpus"],
                )
                ax.plot(
                    [r["gpus"] for r in selected],
                    [r[metric] for r in selected],
                    marker=marker,
                    label=NAMES[backend],
                    color=COLORS[backend],
                    lw=2,
                    markersize=4,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            ax.set_title(title, fontsize=11)
            ax.tick_params(labelsize=10)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        fig.supylabel("GPU time (ms)", fontsize=12)
        fig.supxlabel("GPUs", fontsize=12)
        fig.legend(
            *axs[0, 0].get_legend_handles_labels(),
            loc="outside upper center",
            ncol=3,
            frameon=False,
            fontsize=11,
        )
        save(fig, "scaling_" + op)


def final_profile(data: dict) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.6, 5.75),
        layout="constrained",
        sharex=True,
        gridspec_kw={"height_ratios": [7, 8]},
    )
    convolution = {
        "Direct convolution",
        "Fused spatial convolution",
        "Winograd GEMMs",
        "Winograd transforms",
    }
    for ax, backend in zip(axes, ["native", "winovae"]):
        row = data["b300_decoder_profile"]["results"][backend]
        cats = row["kernel_seconds"]
        total = sum(cats.values())
        for i, (label, seconds) in enumerate(cats.items()):
            ax.barh(i, seconds, height=0.65, color=GREEN if label in convolution else BLUE)
            duration = f"{seconds * 1000:.1f} ms" if seconds < 0.01 else f"{seconds:.2f} s"
            share = f"{seconds / total:.0%}" if seconds / total >= 0.005 else "<1%"
            ax.text(seconds + 0.025, i, f"{duration} ({share})", va="center", fontsize=10)
        ax.set_yticks(range(len(cats)), cats.keys(), fontsize=11)
        ax.invert_yaxis()
        ax.set_title(
            f"{NAMES[backend]}   ·   {row['median_span_s']:.2f} s per decode",
            loc="left",
            fontsize=13,
        )
        ax.tick_params(length=0, labelsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "left", "right"]].set_visible(False)
    axes[-1].set(xlim=(0, 2.55), xticks=[0, 0.5, 1.0, 1.5, 2.0], xlabel="Kernel time (seconds)")
    save(fig, "final_profile")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "text.color": INK,
            "axes.labelcolor": INK,
            "svg.fonttype": "path",
        }
    )
    data = json.loads((ROOT / "data.json").read_text())
    decoder_graph()
    transform_stages()
    causal_first_frame()
    decoder_breakdown(data)
    hardware(data, "hardware_b300")
    hardware(data, "hardware_h200")
    scaling(data)
    final_profile(data)


if __name__ == "__main__":
    main()
