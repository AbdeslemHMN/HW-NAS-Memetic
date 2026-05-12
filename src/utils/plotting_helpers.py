"""Reusable plotting utilities for HW-NAS-Memetic notebooks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def setup_matplotlib_style(dpi: int = 150, font_size: int = 11) -> None:
    plt.rcParams.update({
        "figure.dpi": dpi,
        "font.size": font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def create_grid_figure(rows: int, cols: int, figsize: tuple[float, float]) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    return fig, axes


def make_algo_legend_handles(algo_styles: dict[str, tuple[str, str, str, float, int, float]], order: list[str]) -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [], [],
            color=algo_styles[label][0],
            linestyle=algo_styles[label][1],
            linewidth=algo_styles[label][3],
            marker=algo_styles[label][2],
            markersize=7,
            label=label,
        )
        for label in order
    ]


def style_pareto_panel(
    ax: plt.Axes,
    x_lo: float,
    x_hi: float,
    y_hi: float,
    xlabel: str = "Accuracy (%) ↑",
    ylabel: str = "Latency (ms) ↓",
    title: str | None = None,
    show_xticklabels: bool = True,
    show_yticklabels: bool = True,
) -> None:
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.0, y_hi)
    ax.tick_params(labelsize=8)
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=10)
    if show_xticklabels:
        ax.set_xlabel(xlabel, fontsize=9)
    else:
        ax.set_xticklabels([])
    if show_yticklabels:
        ax.set_ylabel(ylabel, fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.grid(True, alpha=0.2)


def annotate_better_tradeoff(ax: plt.Axes, x_lo: float, x_span: float, y_hi: float) -> None:
    ax.annotate(
        "",
        xy=(x_lo + x_span * 0.80, y_hi * 0.13),
        xytext=(x_lo + x_span * 0.62, y_hi * 0.42),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#999999", alpha=0.7),
        zorder=6,
    )
    ax.text(
        x_lo + x_span * 0.62,
        y_hi * 0.47,
        "Better\ntrade-off",
        color="#999999",
        fontsize=8.5,
        ha="center",
        va="bottom",
        alpha=0.9,
    )


def scatter_raw_points(
    ax: plt.Axes,
    pts: np.ndarray,
    color: str,
    max_points: int = 3000,
    seed: int = 42,
) -> None:
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    ax.scatter(
        -pts[:, 0], pts[:, 1],
        color=color,
        alpha=0.10,
        s=5,
        marker='.',
        linewidths=0,
        zorder=0,
        rasterized=True,
    )


def plot_algorithm_front(
    ax: plt.Axes,
    front: np.ndarray,
    style: tuple[str, str, str, float, int, float],
    label: str | None = None,
) -> mlines.Line2D:
    color, linestyle, marker, linewidth, zorder, alpha = style
    srt = front[np.argsort(front[:, 0])]
    ax.step(
        srt[:, 0], srt[:, 1], where='post',
        color=color, linestyle=linestyle, linewidth=linewidth,
        alpha=alpha, zorder=zorder,
    )
    ax.scatter(
        srt[:, 0], srt[:, 1],
        color=color,
        edgecolors='white',
        marker=marker,
        s=70,
        linewidths=0.9,
        zorder=zorder + 1,
    )
    return mlines.Line2D(
        [], [], color=color, linestyle=linestyle,
        linewidth=linewidth, marker=marker, markersize=7,
        label=label,
    )


def fill_dominated_area(
    ax: plt.Axes,
    fp: np.ndarray,
    fn: np.ndarray,
    step_interp: callable,
) -> None:
    if len(fp) < 2 or len(fn) < 2:
        return
    srt_p = fp[np.argsort(fp[:, 0])]
    srt_n = fn[np.argsort(fn[:, 0])]
    x_fill = np.linspace(
        max(srt_p[0, 0], srt_n[0, 0]),
        min(srt_p[-1, 0], srt_n[-1, 0]),
        600,
    )
    lat_p = step_interp(srt_p[:, 0], srt_p[:, 1], x_fill)
    lat_n = step_interp(srt_n[:, 0], srt_n[:, 1], x_fill)
    better = lat_p < lat_n
    if better.any():
        ax.fill_between(
            x_fill, lat_p, lat_n,
            where=better,
            color="#1f77b4",
            alpha=0.13,
            zorder=3,
            label="_nolegend_",
        )


def draw_knee_inset(
    ax: plt.Axes,
    fp: np.ndarray,
    fronts: dict[str, np.ndarray],
    plot_order: list[str],
    algo_styles: dict[str, tuple[str, str, str, float, int, float]],
    find_knee: callable,
    step_interp: callable,
    mark_inset: callable,
    x_lo: float,
    x_hi: float,
    y_hi: float,
) -> None:
    if len(fp) < 2:
        return
    srt_p = fp[np.argsort(fp[:, 0])]
    ki = find_knee(srt_p)
    knee_acc, knee_lat = float(srt_p[ki, 0]), float(srt_p[ki, 1])
    inset_dx = max((x_hi - x_lo) * 0.06, 0.3)
    inset_dy = max(y_hi * 0.07, 0.3)
    ins_x0 = max(x_lo, knee_acc - inset_dx)
    ins_x1 = min(x_hi, knee_acc + inset_dx)
    ins_y0 = max(0.0, knee_lat - inset_dy)
    ins_y1 = min(y_hi, knee_lat + inset_dy)
    ins_pos = [0.57, 0.54, 0.38, 0.38] if knee_acc < x_lo + (x_hi - x_lo) * 0.5 else [0.05, 0.54, 0.38, 0.38]
    loc1, loc2 = (1, 3) if knee_acc < x_lo + (x_hi - x_lo) * 0.5 else (2, 4)
    axins = ax.inset_axes(ins_pos)
    for algo in plot_order:
        fr = fronts.get(algo)
        if fr is None or not len(fr):
            continue
        style = algo_styles[algo]
        plot_algorithm_front(axins, fr, style)
    axins.set_xlim(ins_x0, ins_x1)
    axins.set_ylim(ins_y0, ins_y1)
    axins.tick_params(labelsize=6.5)
    axins.set_xlabel("Acc (%)", fontsize=6.5)
    axins.set_ylabel("Lat (ms)", fontsize=6.5)
    axins.grid(True, alpha=0.3, linestyle=':')
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="black", alpha=0.45, lw=0.9)


def draw_combo_panel(
    ax: plt.Axes,
    dataset: str,
    hardware: str,
    sets: dict[str, list[list[dict]]],
    aggregate_front: callable,
    focus_range: callable,
    archive_to_points: callable,
    plot_order: list[str],
    algo_styles: dict[str, tuple[str, str, str, float, int, float]],
    step_interp: callable,
    find_knee: callable,
) -> tuple[list[mlines.Line2D], float, float, float]:
    if all(len(v) == 0 for v in sets.values()):
        return [], 0.0, 0.0, 0.0

    fronts = {algo: aggregate_front(sets[algo]) for algo in plot_order if sets.get(algo)}
    x_lo, x_hi, y_hi = focus_range(sets)
    x_span = x_hi - x_lo

    legend_handles: list[mlines.Line2D] = []
    for algo in plot_order:
        archives = sets.get(algo, [])
        if not archives:
            continue

        style = algo_styles[algo]
        color = style[0]
        points = np.vstack([archive_to_points(a) for a in archives])
        scatter_raw_points(ax, points, color)

        front = fronts.get(algo)
        if front is not None and len(front):
            legend_handles.append(plot_algorithm_front(ax, front, style, label=algo))

    fill_dominated_area(
        ax,
        fronts.get("Proposed", np.empty((0, 2))),
        fronts.get("NSGA-II", np.empty((0, 2))),
        step_interp,
    )
    style_pareto_panel(
        ax, x_lo, x_hi, y_hi,
        title=f"Pareto Front — {dataset} / {hardware.replace('_', ' ').title()}",
    )
    annotate_better_tradeoff(ax, x_lo, x_span, y_hi)
    draw_knee_inset(
        ax,
        fronts.get("Proposed", np.empty((0, 2))),
        fronts,
        plot_order,
        algo_styles,
        find_knee,
        step_interp,
        mark_inset,
        x_lo,
        x_hi,
        y_hi,
    )
    return legend_handles, x_lo, x_hi, y_hi


def save_combo_plot(
    dataset: str,
    hardware: str,
    sets: dict[str, list[list[dict]]],
    aggregate_front: callable,
    focus_range: callable,
    archive_to_points: callable,
    results_dir: Path,
    plots_dir: Path,
    plot_order: list[str],
    algo_styles: dict[str, tuple[str, str, str, float, int, float]],
    step_interp: callable,
    find_knee: callable,
) -> bool:
    if all(len(v) == 0 for v in sets.values()):
        print(f"Skipping {dataset}/{hardware}: no archives found.")
        return False

    fronts = {algo: aggregate_front(sets[algo]) for algo in plot_order if sets.get(algo)}
    x_lo, x_hi, y_hi = focus_range(sets)
    x_span = x_hi - x_lo

    fig, ax = plt.subplots(figsize=(8, 5.5))
    legend_handles = []

    for algo in plot_order:
        archives = sets.get(algo, [])
        if not archives:
            continue
        style = algo_styles[algo]
        color = style[0]
        scatter_raw_points(ax, np.vstack([archive_to_points(a) for a in archives]), color)
        front = fronts.get(algo)
        if front is not None and len(front):
            legend_handles.append(plot_algorithm_front(ax, front, style, label=algo))

    fill_dominated_area(ax, fronts.get("Proposed", np.empty((0, 2))), fronts.get("NSGA-II", np.empty((0, 2))), step_interp)
    style_pareto_panel(ax, x_lo, x_hi, y_hi, title=f"Pareto Front — {dataset} / {hardware.replace('_', ' ').title()}")
    annotate_better_tradeoff(ax, x_lo, x_span, y_hi)
    draw_knee_inset(ax, fronts.get("Proposed", np.empty((0, 2))), fronts, plot_order, algo_styles, find_knee, step_interp, mark_inset, x_lo, x_hi, y_hi)

    ax.legend(handles=legend_handles[::-1], title="Algorithm", title_fontsize=10, frameon=True, fontsize=10, labelspacing=0.25, handletextpad=0.4)
    fig.tight_layout(pad=1.0)

    combo_dir = plots_dir / dataset / hardware
    combo_dir.mkdir(parents=True, exist_ok=True)
    out_path = combo_dir / "pareto.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved individual plot → {out_path}")
    return True
