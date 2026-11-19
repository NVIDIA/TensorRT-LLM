# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render article-width WinoVAE figures from the adjacent recorded data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import overview_plots
from matplotlib.patches import Rectangle
from overview_plots import save

ROOT = Path(__file__).resolve().parent
BLUE, GREEN, ORANGE, DARK = "#2563A6", "#507C0C", "#BD5B16", "#203044"
PALE_BLUE, PALE_GREEN, PALE_ORANGE = "#DCE9F7", "#E3EFD3", "#FBE5D3"
GREY, PALE_GREY = "#697586", "#F1F3F5"


def canvas(height: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.4, height))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 6.4), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def text(ax: plt.Axes, x: float, y: float, value: str, **kwargs) -> None:
    ax.text(x, y, value, ha="center", va="center", color=DARK, fontsize=16, **kwargs)


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, value: str, color: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="white", linewidth=2))
    if value:
        text(ax, x + w / 2, y + h / 2, value)


def arrow(ax: plt.Axes, start: tuple, end: tuple, color: str = GREY) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "lw": 2})


def clean(ax: plt.Axes, direction: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=direction, color="#E3E7EC")
    ax.tick_params(length=0, pad=8)


def flow() -> None:
    fig, ax = canvas(1.68)
    left, step = 0.84, 0.54
    for j in range(6):
        text(ax, left + (j + 0.5) * step, 1.43, rf"$\mathbf{{x}}_{j}$")
    text(ax, 5.40, 1.43, r"$V_0$")

    for tile, first, y, fill, ink in [
        ("A", 0, 0.75, PALE_GREEN, GREEN),
        ("B", 2, 0.17, PALE_ORANGE, ORANGE),
    ]:
        center = y + 0.24
        ax.text(0.70, center, f"Tile {tile}", ha="right", va="center", color=ink, fontsize=14)
        for offset, weight in enumerate(("+1", "0", "−1", "0")):
            x = left + (first + offset) * step
            box(ax, x, y, step, 0.48, weight, fill)
        arrow(ax, (4.16, center), (4.50, center))
        row = rf"$\mathbf{{x}}_{first} - \mathbf{{x}}_{first + 2}$"
        box(ax, 4.63, y, 1.54, 0.48, row, fill)
    save(fig, "channel_flow")


def tiles() -> None:
    fig, ax = canvas(2.78)
    text(ax, 3.2, 2.54, r"$F(h\times w,\;h_f\times w_f)$")
    for x, label in [(1.4, "Input tile"), (3.45, "Filter"), (5.45, "Output tile")]:
        text(ax, x, 2.0, label)
    for x, y, width, height, color, height_label, width_label in [
        (0.55, 0.4, 1.7, 1.25, PALE_BLUE, r"$h+h_f-1$", r"$w+w_f-1$"),
        (3.05, 0.65, 0.8, 0.75, PALE_ORANGE, r"$h_f$", r"$w_f$"),
        (4.9, 0.65, 1.1, 0.75, PALE_GREEN, r"$h$", r"$w$"),
    ]:
        box(ax, x, y, width, height, "", color)
        text(ax, x - 0.2, y + height / 2, height_label, rotation=90)
        text(ax, x + width / 2, y - 0.23, width_label)
    text(ax, 2.55, 1.02, "∗")
    arrow(ax, (4.02, 1.02), (4.46, 1.02))
    save(fig, "spatial_tiles")


def gemm() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.85))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 9.4), ylim=(0, 5.85))
    ax.axis("off")
    for x, size, label, color in [
        (0.50, 4, "Input tiles", PALE_GREEN),
        (5.15, 3, "Filters", PALE_BLUE),
    ]:
        text(ax, x + 0.36, 5.58, label)
        for row in range(size):
            for column in range(size):
                box(
                    ax,
                    x + column * 0.72 / size,
                    4.72 + row * 0.72 / size,
                    0.72 / size,
                    0.72 / size,
                    "",
                    color,
                )
    arrow(ax, (1.38, 5.08), (3.60, 5.08))
    text(ax, 2.49, 5.41, "Input transform")
    text(ax, 4.22, 5.08, r"$V_0,\ldots,V_{15}$")
    arrow(ax, (6.03, 5.08), (7.38, 5.08))
    text(ax, 6.705, 5.51, "Filter\ntransform")
    text(ax, 8.06, 5.08, r"$U_0,\ldots,U_{15}$")
    text(ax, 2.45, 4.58, r"$M_i=V_iU_i$")
    specs = [(0.15, 1.35, "V"), (1.88, 1.10, "U"), (3.40, 1.35, "M")]
    for x, label in [
        (0.825, r"$N_{\mathrm{tiles}}\times C_{\mathrm{in}}$"),
        (2.43, r"$C_{\mathrm{in}}\times C_{\mathrm{out}}$"),
        (4.075, r"$N_{\mathrm{tiles}}\times C_{\mathrm{out}}$"),
    ]:
        ax.text(x, 4.25, label, ha="center", va="center", color=DARK, fontsize=14)
    for coordinate, y in [(0, 2.95), (1, 1.60), (15, 0.05)]:
        for x, width, name in specs:
            text(ax, x + width / 2, y + 1.03, rf"${name}_{{{coordinate}}}$")
            if name == "U":
                box(ax, x, y, width, 0.84, "filters", PALE_BLUE)
            else:
                box(ax, x, y + 0.56, width, 0.28, "tile A", PALE_GREEN)
                box(ax, x, y + 0.28, width, 0.28, "tile B", PALE_ORANGE)
                box(ax, x, y, width, 0.28, "…", PALE_GREY)
        text(ax, 3.19, y + 0.42, "=")
        ax.plot([4.75, 5.12], [y + 0.70, y + 0.70], color=GREEN, lw=2)
    for x in [0.825, 2.43, 4.075]:
        text(ax, x, 1.33, "⋮")
    ax.plot([5.12, 5.12], [0.75, 3.65], color=GREEN, lw=2)
    arrow(ax, (5.12, 2.30), (5.57, 2.30), GREEN)
    text(ax, 6.23, 3.95, "tile A")
    for coordinate in range(16):
        y = 3.66 - (coordinate + 1) * 0.17
        ax.add_patch(
            Rectangle((5.62, y), 1.22, 0.17, facecolor=PALE_GREEN, edgecolor=GREEN, linewidth=0.5)
        )
        ax.text(5.70, y + 0.085, str(coordinate), ha="left", va="center", color=GREEN, fontsize=10)
    text(ax, 6.23, 0.62, r"$16\times C_{\mathrm{out}}$")
    arrow(ax, (6.87, 2.30), (8.18, 2.30), GREEN)
    text(ax, 7.525, 2.74, "Output\ntransform")
    for row in range(2):
        for column in range(2):
            box(ax, 8.23 + column * 0.36, 1.94 + row * 0.36, 0.36, 0.36, "", PALE_GREEN)
    text(ax, 8.59, 3.10, "Output\ntile A")
    text(ax, 8.59, 1.43, "2 × 2 per\noutput channel")
    save(fig, "coordinate_gemm")


def fusion() -> None:
    fig, ax = canvas(5.25)
    text(ax, 1.55, 4.98, "Separate kernels", weight="bold")
    text(ax, 4.85, 4.98, "Fused kernel", weight="bold")
    text(ax, 3.2, 4.57, "Read convolution output, bias and residual")
    box(ax, 3.4, 1.15, 2.9, 3.05, "", PALE_GREEN)
    for y, label in [(3.65, "Bias + residual"), (2.4, "Normalize"), (1.15, "SiLU + transform")]:
        box(ax, 0.1, y, 2.9, 0.55, label, PALE_BLUE)
        text(ax, 4.85, y + 0.275, label)
    for y in [3.06, 1.81]:
        box(ax, 0.1, y, 2.9, 0.38, "HBM temporary", PALE_ORANGE)
        arrow(ax, (1.55, y + 0.59), (1.55, y + 0.39))
        arrow(ax, (1.55, y - 0.02), (1.55, y - 0.11))
        arrow(ax, (4.85, y + 0.53), (4.85, y - 0.1))
    for x in [1.55, 4.85]:
        arrow(ax, (x, 4.43), (x, 4.22))
        arrow(ax, (x, 1.11), (x, 0.52))
    box(ax, 0.1, 0.02, 6.2, 0.48, "Write the convolution input to HBM", PALE_GREY)
    save(fig, "fusion")


def upsample_fold() -> None:
    fig, ax = canvas(4.18)
    colors = [PALE_BLUE, PALE_ORANGE, PALE_GREEN, "#E8DFF4"]

    def grid(
        x: float,
        y: float,
        values: list[list[str]],
        groups: list[list[int]],
        width: float,
        height: float,
    ) -> None:
        for row, entries in enumerate(values):
            for col, value in enumerate(entries):
                box(
                    ax,
                    x + col * width,
                    y + (len(values) - 1 - row) * height,
                    width,
                    height,
                    value,
                    colors[groups[row][col]],
                )

    for y, label in [(3.95, "Upsampled input"), (1.55, "Original input")]:
        text(ax, 1.25, y, label)
    text(ax, 4.24, 3.95, "3 × 3 filter")
    text(ax, 4.24, 1.55, "Folded 2 × 2 filter")

    repeated = [
        ["A", "A", "B", "B"],
        ["A", "A", "B", "B"],
        ["C", "C", "D", "D"],
        ["C", "C", "D", "D"],
    ]
    groups = [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]
    grid(0.41, 1.98, repeated, groups, 0.42, 0.42)
    ax.add_patch(Rectangle((0.41, 2.40), 1.26, 1.26, fill=False, edgecolor=GREEN, lw=2, ls="--"))
    grid(
        3.47,
        2.06,
        [[rf"$w_{3 * r + c}$" for c in range(3)] for r in range(3)],
        [[0, 0, 1], [0, 0, 1], [2, 2, 3]],
        0.51,
        0.51,
    )

    grid(0.70, 0.12, [["A", "B"], ["C", "D"]], [[0, 1], [2, 3]], 0.55, 0.55)
    folded = [["$w_0+w_1$\n$+w_3+w_4$", "$w_2+w_5$"], ["$w_6+w_7$", "$w_8$"]]
    grid(3.04, 0.09, folded, [[0, 1], [2, 3]], 1.2, 0.58)
    for middle in [2.83, 0.67]:
        text(ax, 2.55, middle, r"$\ast$")
        arrow(ax, (5.48, middle), (5.90, middle))
        text(ax, 6.10, middle, "$y$")
    save(fig, "upsample_fold")


def frame_strip(
    ax: plt.Axes,
    x: float,
    y: float,
    columns: range,
    step: float,
    *,
    halos: tuple[int, ...] = (),
    labels_above: bool = False,
) -> None:
    # The same contour and column numbers identify a strip in every buffer.
    ridge = [0.22, 0.34, 0.71, 0.43, 0.31, 0.57, 0.77, 0.46, 0.26]
    for i, j in enumerate(columns):
        left = x + i * step
        if j in (-1, 8):
            ax.add_patch(Rectangle((left, y), step, 0.85, facecolor=PALE_GREY, edgecolor="white"))
            ax.text(left + step / 2, y + 0.425, "0", ha="center", va="center", color=GREY)
            continue
        fill, edge = (PALE_BLUE, BLUE) if j < 4 else (PALE_GREEN, GREEN)
        ax.add_patch(Rectangle((left, y), step, 0.85, facecolor=fill, edgecolor="white"))
        ax.fill_between(
            [left, left + step],
            y,
            [y + ridge[j], y + ridge[j + 1]],
            color=edge,
            alpha=0.60,
            linewidth=0,
        )
        if j in halos:
            ax.add_patch(
                Rectangle((left, y), step, 0.85, fill=False, edgecolor=edge, ls="--", lw=1.4)
            )
        ax.text(
            left + step / 2,
            y + 0.94 if labels_above else y - 0.12,
            str(j),
            ha="center",
            va="bottom" if labels_above else "top",
            fontsize=11,
        )


def halo() -> None:
    fig, ax = canvas(3.55)
    step = 0.45
    origin = 1.75
    for rank, first, last, y, halo_column in [(0, -1, 4, 2.35, 4), (1, 3, 8, 0.35, 3)]:
        ax.text(0.78, y + 0.43, f"GPU {rank}", va="center", ha="right", fontsize=14)
        frame_strip(
            ax,
            origin + first * step,
            y,
            range(first, last + 1),
            step,
            halos=(halo_column,),
            labels_above=rank == 0,
        )
    for j, start, end, color in [(3, 2.35, 1.20, BLUE), (4, 1.20, 2.35, GREEN)]:
        x = origin + (j + 0.5) * step
        arrow(ax, (x, start), (x, end), color)
    save(fig, "halo_exchange")

    fig, ax = canvas(3.15)
    step = 0.36
    ax.text(1.82, 2.88, "Local strip", ha="center", fontsize=14)
    ax.text(4.83, 2.88, "After all-gather", ha="center", fontsize=14)
    for rank, y in [(0, 1.66), (1, 0.25)]:
        ax.text(0.87, y + 0.43, f"GPU {rank}", va="center", ha="right", fontsize=14)
        frame_strip(ax, 1.1, y, range(4 * rank, 4 * rank + 4), step)
        arrow(ax, (2.62, y + 0.425), (3.28, y + 0.425))
        frame_strip(ax, 3.39, y, range(8), step)
    save(fig, "strip_allgather")


def costs(data: dict) -> None:
    rows = data["spatial_cost_model"]["rows"]
    xs = np.arange(len(rows))
    labels = ["Direct"] + [f"F({m}×{m})" for m in range(2, 9)]
    for name, key, title, unit in [
        ("cost_flops", "gemm_flops_per_output_pixel", "Arithmetic per output pixel", "FLOPs"),
        (
            "cost_workspace",
            "extra_workspace_bytes_per_output_pixel",
            "Extra V + M workspace per output pixel",
            "KiB",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 3.7), layout="constrained")
        values = np.array([r[key] for r in rows])
        if unit == "KiB":
            values = values / 1024
        ax.plot(xs[:1], values[:1], "o", color=ORANGE)
        ax.plot(xs[1:], values[1:], "o-", color=ORANGE, lw=2.3)
        ax.set_title(title, loc="left", fontsize=17, pad=16)
        if unit == "FLOPs":
            simt = [r["transform_ops_per_output_pixel"] for r in rows]
            ax.plot(xs[1:], simt[1:], "s-", color=BLUE, lw=2.3)
            ax.set_yscale("log")
            ax.set_ylim(7e3, 5e7)
            ax.set_yticks([1e4, 1e5, 1e6, 1e7], ["10k", "100k", "1M", "10M"])
            ax.text(3, 1.3e7, "GEMM", color=ORANGE)
            ax.text(2.5, 1.1e5, "Transforms", color=BLUE)
        else:
            ax.set_ylim(-0.8, 20)
            ax.set_yticks([0, 4, 8, 12, 16])
            for i in [1, 3, 7]:
                ax.annotate(
                    f"{values[i]:.0f}",
                    (i, values[i]),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=16,
                )
        ax.set_xticks(xs, labels, rotation=40, ha="right", fontsize=13)
        ax.set_xlabel("Spatial output tile (3 × 3 filter)", fontsize=14)
        ax.set_ylabel(unit)
        clean(ax, "y")
        save(fig, name)


def sweep(data: dict) -> None:
    rows = data["spatial_sweep"]["layers"]
    methods = [
        ("F(2×2) cuBLAS", "2D F(2 × 2, 3 × 3), cuBLAS", BLUE),
        ("F(4×4) KF", "2D F(4 × 4, 3 × 3), specialized", ORANGE),
        ("fused 1D F(2,3)", "1D F(2, 3) in width + direct height, fused", GREEN),
    ]
    fig, axs = plt.subplots(3, 1, figsize=(6.4, 8.7), layout="constrained", sharex=True)
    for ax, channels in zip(axs, [1024, 512, 256]):
        group = [r for r in rows if r["channels"] == channels]
        for key, label, color in methods:
            points = [
                (i, r["span_us"]["direct cuDNN"] / r["span_us"][key])
                for i, r in enumerate(group)
                if key in r["span_us"]
            ]
            if points:
                x, y = zip(*points)
                style = {"F(2×2) cuBLAS": "o-", "F(4×4) KF": "^-", "fused 1D F(2,3)": "s:"}[key]
                ax.plot(x, y, style, label=label, color=color, lw=2.3)
        ax.axhline(1, color=GREY, ls="--", lw=1.5)
        ax.set(ylim=(0, 2.7), yticks=[0, 1, 2])
        ax.set_title(f"{channels} channels", loc="left", fontsize=17)
        clean(ax, "y")
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", frameon=False, fontsize=13)
    axs[1].set_ylabel("Speedup over direct cuDNN")
    sizes = [(r["height"], r["width"]) for r in rows if r["channels"] == 256]
    axs[-1].set_xticks(
        range(len(sizes)), [f"{h}×{w}" for h, w in sizes], rotation=40, ha="right", fontsize=12
    )
    axs[-1].set_xlabel("Feature-frame size")
    save(fig, "cudnn_sweep")


def halo_latency(data: dict) -> None:
    values = np.array(data["halo_exchange"]["width_split"])
    fig, ax = plt.subplots(figsize=(6.4, 4.7), layout="constrained")
    y = np.arange(4)
    ax.barh(y - 0.18, values[:, 0], height=0.32, color=GREY, label="NCCL")
    ax.barh(y + 0.18, values[:, 1], height=0.32, color=GREEN, label="Fused GPUDirect P2P")
    for i in range(4):
        for j, offset in [(0, -0.18), (1, 0.18)]:
            ax.text(values[i, j] + 6, i + offset, f"{values[i, j]:.0f}", va="center", fontsize=15)
    ax.set_yticks(y, ["720p\ndecode", "720p\nencode", "480p\ndecode", "480p\nencode"])
    ax.set(xlim=(0, 570), xlabel="Whole-call latency (ms)")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right", fontsize=15)
    clean(ax)
    save(fig, "halo_latency")


def motivation(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.45, 4.25))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.98, bottom=0.23)
    values = data["motivation"]["seconds"]
    total = sum(values)
    labels = ["VAE decode", "Transformer rollout", "Prompt + VAE encode"]
    wedges, _ = ax.pie(
        values,
        colors=["#76B900", "#4C5B6B", "#9AA7B3"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 3},
        explode=[0.04, 0, 0],
    )
    ax.legend(
        wedges,
        [
            f"{label}: {value:.1f} s ({100 * value / total:.0f}%)"
            for label, value in zip(labels, values)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=13,
        handlelength=1.2,
    )
    ax.text(
        0,
        0.07,
        f"{total:.1f} s",
        ha="center",
        va="center",
        fontsize=23,
        weight="bold",
        color="#1F2933",
    )
    ax.text(
        0,
        -0.17,
        f"per {data['motivation']['frames']}-frame clip",
        ha="center",
        va="center",
        fontsize=11,
        color="#5F6B76",
    )
    save(fig, "motivation")


def bars(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.5), layout="constrained")
    values = np.array(data["latency"]["median_ms"]) / 1000
    ax.barh(["BF16\nWan VAE", "WinoVAE"], values, color=[BLUE, GREEN], height=0.5)
    for i, value in enumerate(values):
        ax.text(value + 0.12, i, f"{value:.2f} s", va="center")
    ax.invert_yaxis()
    ax.set(xlim=(0, 7.6), xticks=[0, 2, 4, 6], xlabel="Decode latency (seconds)")
    clean(ax)
    save(fig, "latency")
    methods = data["nvfp4"]["methods"]
    values = np.array([row["errors"] for row in methods]) * 100
    means = values.mean(axis=1)
    errors = np.array([means - values.min(axis=1), values.max(axis=1) - means])
    fig, ax = plt.subplots(figsize=(6.4, 3.9), layout="constrained")
    ax.barh(
        [
            "Direct",
            "Winograd\n{0, 1, −1, 2, −½, ∞}",
            "Winograd\nsearched points",
            "Winograd\nimplemented",
            "Fourier68",
        ],
        means,
        xerr=errors,
        color=[BLUE, ORANGE, ORANGE, ORANGE, GREEN],
        capsize=3,
        height=0.6,
    )
    for i, value in enumerate(means):
        ax.text(value + 2, i, f"{value:.1f}%", va="center", fontsize=15)
    ax.invert_yaxis()
    ax.set(xlim=(0, 121), xticks=[0, 25, 50, 75, 100], xlabel="Relative-L2 error (%)")
    clean(ax)
    save(fig, "nvfp4_error")


def main() -> None:
    data = json.loads((ROOT / "data.json").read_text())
    plt.rcParams.update(
        {
            "font.size": 16,
            "font.family": "DejaVu Sans",
            "text.color": DARK,
            "axes.labelcolor": DARK,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "svg.fonttype": "path",
        }
    )
    flow()
    tiles()
    gemm()
    fusion()
    upsample_fold()
    halo()
    costs(data)
    sweep(data)
    halo_latency(data)
    motivation(data)
    bars(data)
    overview_plots.main()


if __name__ == "__main__":
    main()
