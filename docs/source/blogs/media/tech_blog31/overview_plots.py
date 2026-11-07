# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render decoder structure and measured WinoVAE workload profiles."""

import json
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
    target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
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
                ax.text(
                    x + 0.45, tile_y - 0.25, "Term vanishes", ha="center", fontsize=10, color=GREY
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
        ax.text(
            3.2,
            line_y - 0.31,
            "One spatial convolution" if first else "Three spatial convolutions, summed",
            ha="center",
            fontsize=12,
            color=INK,
        )
    save(fig, "causal_first_frame")


def transform_stages() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.set(xlim=(0, 6.4), ylim=(0, 4.3))
    ax.axis("off")
    node(ax, 0.1, 3.15, 2.5, 0.8, "Input transform\nv = Bᵀd", "#DCE9F7", 15)
    node(ax, 3.8, 3.15, 2.5, 0.8, "Filter transform\nu = Gg", "#FBE5D3", 15)
    ax.text(1.35, 2.82, "Per tile · SIMT", ha="center", fontsize=13, color=BLUE)
    ax.text(5.05, 2.82, "Once at weight loading", ha="center", fontsize=13, color="#BD5B16")
    node(ax, 1.55, 1.55, 3.3, 0.76, "Products: mᵢ = vᵢuᵢ", "#E3EFD3", 15)
    arrow(ax, (1.35, 2.65), (2.4, 2.35))
    arrow(ax, (5.05, 2.65), (4.0, 2.35))
    ax.text(3.2, 1.23, "With channels: batched GEMM on Tensor Cores", ha="center", fontsize=12)
    node(ax, 1.55, 0.05, 3.3, 0.65, "Output transform\ny = Aᵀm", "#EAF0F6", 14)
    arrow(ax, (3.2, 1.10), (3.2, 0.75))
    ax.text(5.05, 0.35, "Per tile\nSIMT", fontsize=12, va="center", color=GREY)
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
    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    ax.barh(labels, vals, color=["#76B900"] + ["#B5DC6A"] * 5, height=0.64)
    for i, v in enumerate(vals):
        ax.text(v + 0.08, i, f"{v:.1f} s ({100 * v / sum(vals):.0f}%)", va="center", fontsize=12)
    ax.invert_yaxis()
    ax.set(xlim=(0, 6.3), xlabel="Decoder time (seconds)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=12)
    save(fig, "decoder_breakdown")


def hardware(data: dict, key: str) -> None:
    rows = data[key]["results"]
    fig, axs = plt.subplots(4, 2, figsize=(6.4, 9.0), layout="constrained")
    for i, (shape, title) in enumerate(WORKLOADS):
        for j, op in enumerate(["encode", "decode"]):
            ax = axs[i, j]
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
            ax.set_ylabel("GPU time (ms)", fontsize=11)
            ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=7)
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
        fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.9), layout="constrained")
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
            for backend in NAMES:
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
                    "o-",
                    color=COLORS[backend],
                    lw=2,
                    markersize=4,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("GPUs", fontsize=11)
            ax.set_ylabel("GPU time (ms)", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        fig.legend(
            [Patch(color=COLORS[b]) for b in NAMES],
            list(NAMES.values()),
            loc="outside lower center",
            ncol=3,
            frameon=False,
            fontsize=11,
        )
        save(fig, "scaling_" + op)


def final_profile(data: dict) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 7.2))
    palettes = [
        ["#76B900", "#2F5D8F", "#7FA7D6", "#B7CDE8", "#D6E3F3", "#1F2937", "#D1D5DB"],
        ["#3E6B00", "#76B900", "#A6D65A", "#D3EBAE", "#2F5D8F", "#1F2937", "#B7CDE8", "#D1D5DB"],
    ]
    for ax, backend, colors in zip(axs, ["native", "winovae"], palettes):
        row = data["b300_decoder_profile"]["results"][backend]
        cats = row["kernel_seconds"]
        vals = list(cats.values())
        wedges, _ = ax.pie(
            vals,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.4, "edgecolor": "white"},
        )
        ax.text(0, 0.06, f"{row['median_span_s']:.2f} s", ha="center", fontsize=21, weight="bold")
        ax.text(0, -0.20, "per decode", ha="center", fontsize=11, color=GREY)
        ax.set_title(NAMES[backend], fontsize=15, weight="bold")
        labels = []
        for label, value in cats.items():
            duration = f"{value:.2f} s" if value >= 0.01 else f"{value * 1000:.1f} ms"
            fraction = f"{100 * value / sum(vals):.0f}%" if value / sum(vals) >= 0.005 else "<1%"
            labels.append(f"{label}\n{duration} · {fraction}")
        ax.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(0.97, 0.5),
            fontsize=10,
            frameon=False,
            labelspacing=0.55,
            handlelength=1,
        )
    fig.subplots_adjust(left=0.0, right=0.48, hspace=0.38, top=0.95, bottom=0.03)
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
