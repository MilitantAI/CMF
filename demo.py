"""
Satellite point detector demo.

C = environmental density gradient along satellite → target throughline.
R = process time t indexing resolution R(t) = (ℓ, τ, Λ, ε, Σ).

Shows: process functional sharpens over time;
       standard fixed functional misses this entirely.

Run: python demo.py [--open]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import os
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import (
    Context,
    MetricCoeffs,
    ObserverThroughline,
    PredictionReport,
    ProcessSnapshot,
    SpacetimeGrid,
    bolted_on_functional,
    config_at_process_time,
    context_gradient_field,
    effective_metric,
    environmental_density,
    evolve_process,
    prediction_report,
    process_functional,
    throughline_parameter,
)


def gaussian_target(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Surface target process field."""
    envelope = np.exp(-((x - 0.0) ** 2 + (t - 0.0) ** 2) / (2 * 1.2**2))
    return envelope * np.exp(1j * 2.0 * np.pi * x)


def plot_throughline_and_context(
    grid: SpacetimeGrid,
    context: Context,
    t: np.ndarray,
    x: np.ndarray,
    out_path: Path,
) -> None:
    rho_env = environmental_density(grid)
    c_field = context_gradient_field(grid, context)
    s = throughline_parameter(grid, context.throughline)
    pos = context.throughline.position
    tgt = context.throughline.target

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("C = environmental density gradient along observer throughline", fontsize=12)

    im0 = axes[0].pcolormesh(x, t, rho_env, shading="auto", cmap="Blues")
    axes[0].plot([pos[1], tgt[1]], [pos[0], tgt[0]], "r-", linewidth=2, label="throughline")
    axes[0].plot(pos[1], pos[0], "r^", markersize=10, label="satellite")
    axes[0].plot(tgt[1], tgt[0], "g*", markersize=12, label="target")
    axes[0].set_title(r"Environmental density $\rho_{\rm env}$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("process time t")
    axes[0].legend(fontsize=7)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].pcolormesh(x, t, c_field, shading="auto", cmap="RdYlBu_r")
    axes[1].plot([pos[1], tgt[1]], [pos[0], tgt[0]], "k-", linewidth=1.5)
    axes[1].set_title(r"Context $C = \hat{n}\cdot\nabla\rho_{\rm env}$ on throughline")
    axes[1].set_xlabel("x")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].plot(s[len(t) // 2, :], c_field[len(t) // 2, :], color="tab:red")
    axes[2].set_xlabel("throughline parameter s")
    axes[2].set_ylabel(r"$C$")
    axes[2].set_title("C along satellite → target ray at t=0")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_process_vs_bolted_on(
    phi: np.ndarray,
    grid: SpacetimeGrid,
    context: Context,
    history: list[ProcessSnapshot],
    coeffs: MetricCoeffs,
    out_path: Path,
) -> None:
    t_proc = [s.process_time for s in history]
    obs = [s.observer_value for s in history]
    residual = [s.residual_integral for s in history]
    h_res = [s.h_residual_mean for s in history]
    capacity = [s.distinction_capacity for s in history]

    fixed = bolted_on_functional(phi, grid)
    fixed_line = [fixed] * len(t_proc)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        "Process functional F[\u03a6; C, R(t)] sharpens over process time\n"
        "vs fixed F[\u03a6] with C, R bolted on as external inputs",
        fontsize=11,
    )

    axes[0, 0].plot(t_proc, obs, "o-", color="tab:blue", label=r"$F[\Phi; C, R(t)]$ process")
    axes[0, 0].axhline(fixed, color="gray", linestyle="--", label=r"fixed $F[\Phi]$ (C,R ignored)")
    axes[0, 0].set_xlabel("process time t")
    axes[0, 0].set_ylabel("observer reading")
    axes[0, 0].set_title("Functional accuracy improves with process time")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(t_proc, residual, color="tab:orange")
    axes[0, 1].set_xlabel("process time t")
    axes[0, 1].set_ylabel(r"$\int|r_{C,R}|^2$")
    axes[0, 1].set_title("Residual falls as process runs")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(t_proc, h_res, color="tab:red")
    axes[1, 0].set_xlabel("process time t")
    axes[1, 0].set_ylabel("residual-driven |h|")
    axes[1, 0].set_title("What flat Minkowski suppressed — gone at late R")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_proc, capacity, "d-", color="tab:green")
    axes[1, 1].set_xlabel("process time t")
    axes[1, 1].set_ylabel("distinction capacity (kernel width)")
    axes[1, 1].set_title("t ↦ R(t): capacity grows with process time")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(
    phi: np.ndarray,
    grid: SpacetimeGrid,
    context: Context,
    history: list[ProcessSnapshot],
    coeffs: MetricCoeffs,
    t: np.ndarray,
    x: np.ndarray,
    out_path: Path,
) -> None:
    early = history[0]
    late = history[-1]
    cfg_early = config_at_process_time(context, early.process_time, coeffs)
    cfg_late = config_at_process_time(context, late.process_time, coeffs)

    g_early = effective_metric(phi, cfg_early, grid)
    g_late = effective_metric(phi, cfg_late, grid)
    gxx_early = g_early[..., 1, 1]
    gxx_late = g_late[..., 1, 1]
    vmin = float(min(gxx_early.min(), gxx_late.min()))
    vmax = float(max(gxx_early.max(), gxx_late.max()))
    ti = len(t) // 2

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Early vs late process time: low-residual limit", fontsize=13)

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(x, t, gxx_early, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax1.set_title(f"Early t={early.process_time:.1f}\nR(t) coarse, high residual")
    ax1.set_ylabel("t")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(x, t, gxx_late, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax2.set_title(f"Late t={late.process_time:.1f}\nR(t) refined, low residual")
    fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.02, label=r"$g_{xx}$")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, np.ones_like(x), "k--", label=r"Minkowski $\eta_{xx}=1$")
    ax3.plot(x, g_early[ti, :, 1, 1], color="tab:orange", label=f"early R={early.process_time:.1f}")
    ax3.plot(x, g_late[ti, :, 1, 1], color="tab:blue", label=f"late R={late.process_time:.1f}")
    ax3.set_xlabel("x")
    ax3.set_ylabel(r"$g_{xx}$")
    ax3.set_title("Slice at t=0")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    text = (
        f"C (fixed throughline gradient):\n"
        f"  satellite at x={context.throughline.position[1]:.0f}\n"
        f"  target at x={context.throughline.target[1]:.0f}\n"
        f"  |C| along line = {early.context_along_line:.3f}\n\n"
        f"Early t={early.process_time:.1f} (R(t) coarse):\n"
        f"  observer = {early.observer_value:.3f}\n"
        f"  |r|^2 = {early.residual_integral:.3e}\n\n"
        f"Late t={late.process_time:.1f} (R(t) refined):\n"
        f"  observer = {late.observer_value:.3f}\n"
        f"  |r|^2 = {late.residual_integral:.3e}\n\n"
        f"Fixed F[Phi] (bolted-on approach):\n"
        f"  always {bolted_on_functional(phi, grid):.3f}\n"
        f"  — same at all R, C not in map"
    )
    ax4.text(0.02, 0.98, text, va="top", family="monospace", fontsize=9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(
    report: PredictionReport,
    out_path: Path,
) -> None:
    t_proc = report.process_times
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Minkowski stipulates · CMF determines · CMF predicts\n"
        "(not raw |Φ|²; cogenetic observables)",
        fontsize=11,
    )

    # P1: CMF determines residual; Minkowski stipulates zero
    axes[0, 0].semilogy(t_proc, report.residuals, "o-", color="tab:orange", label=r"CMF determines $\int|r|^2$")
    axes[0, 0].axhline(report.minkowski_residual_claim + 1e-12, color="gray", linestyle=":",
                       label="Minkowski stipulates $|r|=0$")
    axes[0, 0].set_xlabel("process time t")
    axes[0, 0].set_ylabel("residual")
    axes[0, 0].set_title("P1: residual — determines at each R; falls early→late")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # P4: CMF determines interval correction; Minkowski stipulates g_xx = η_xx
    axes[0, 1].plot(t_proc, report.interval_errors, "o-", color="tab:red",
                    label=r"CMF determines $|g_{xx}-\eta_{xx}|$")
    axes[0, 1].axhline(0, color="gray", linestyle=":", label="Minkowski stipulates zero")
    axes[0, 1].set_xlabel("process time t")
    axes[0, 1].set_ylabel(r"$|g_{xx}-1|$")
    axes[0, 1].set_title("P4 determines: interval correction (stipulated zero)")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # P5: CMF predicts refinement; Minkowski stipulates no R-map
    axes[1, 0].semilogy(
        t_proc,
        [max(e, 1e-12) for e in report.refinement_errors],
        "o-",
        color="tab:blue",
        label=r"CMF predicts $|O(C,R)-O(C,R_{\rm late})|\to 0$",
    )
    axes[1, 0].set_xlabel("process time t")
    axes[1, 0].set_ylabel("refinement error")
    axes[1, 0].set_title("P5 predicts: sharpening (no R-map in Minkowski)")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    # P2: CMF predicts R-dependence; Minkowski stipulates fixed F[Φ]
    axes[1, 1].plot(t_proc, report.cmf_observer, "o-", color="tab:blue", label=r"CMF determines $O(C,R(t))$")
    axes[1, 1].axhline(report.bolted_on_fixed, color="gray", linestyle="--",
                       label=r"Minkowski stipulates fixed $F[\Phi]$")
    axes[1, 1].set_xlabel("process time t")
    axes[1, 1].set_ylabel("functional value")
    axes[1, 1].set_title("P2 predicts: $R$-dependence (stipulated absent)")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    note = (
        f"late R: |r|^2={report.late_residual:.3f}, g_xx={report.late_g_xx_target:.3f}, "
        f"O={report.late_observer:.3f}"
    )
    fig.text(0.5, 0.02, note, ha="center", family="monospace", fontsize=8, color="0.35")

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_demo(*, open_viewer: bool = False) -> Path:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    nt, nx = 128, 256
    t = np.linspace(-4.0, 4.0, nt)
    x = np.linspace(-8.0, 8.0, nx)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    phi = gaussian_target(tt, xx)
    grid = SpacetimeGrid(t=t, spatial=(x,), phi=phi)

    # C: gradient of atmosphere along satellite → target throughline
    context = Context(
        throughline=ObserverThroughline(
            position=(-2.0, -7.0),  # (t, x): satellite at altitude x=-7
            target=(0.0, 0.0),      # looking at surface target
        )
    )
    coeffs = MetricCoeffs()

    history = evolve_process(phi, grid, context, coeffs=coeffs, n_times=24)

    ctx_path = out_dir / "throughline.png"
    proc_path = out_dir / "process.png"
    comp_path = out_dir / "comparison.png"
    pred_path = out_dir / "predictions.png"

    report = prediction_report(phi, grid, context, history)

    plot_throughline_and_context(grid, context, t, x, ctx_path)
    plot_process_vs_bolted_on(phi, grid, context, history, coeffs, proc_path)
    plot_comparison(phi, grid, context, history, coeffs, t, x, comp_path)
    plot_predictions(report, pred_path)

    early, late = history[0], history[-1]
    print(f"C: density gradient along satellite (x=-7) -> target (x=0)")
    print(f"t: process time {early.process_time:.1f} -> {late.process_time:.1f}  (R(t) sharpens)")
    print(f"  stipulates |r|=0, |g_xx-1|=0, fixed F[Phi]={report.bolted_on_fixed:.4f}")
    print(f"  determines |r|^2:  {early.residual_integral:.4e} -> {late.residual_integral:.4e}")
    print(f"  determines |g_xx-1|: {report.interval_errors[0]:.4f} -> {report.interval_errors[-1]:.4f}")
    print(f"  determines O(C,R): {early.observer_value:.4f} -> {late.observer_value:.4f}")
    print(f"  predicts refine:   {report.refinement_errors[0]:.4f} -> {report.refinement_errors[-1]:.4e}")
    print(f"Saved {ctx_path}, {proc_path}, {comp_path}, {pred_path}")

    if open_viewer:
        if sys.platform == "win32":
            os.startfile(comp_path)  # noqa: S606
        else:
            webbrowser.open(comp_path.as_uri())

    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMF satellite point-detector demo")
    parser.add_argument("--open", action="store_true", help="open comparison figure after run")
    args = parser.parse_args()
    run_demo(open_viewer=args.open)
