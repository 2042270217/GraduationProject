"""训练与测试结果可视化：曲线、散点、残差、多步指标等。"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import config


def _finalize(fig: plt.Figure, out_name: str | None) -> None:
    if out_name:
        config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.PLOT_DIR / out_name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print("Saved figure:", path)
    if getattr(config, "SHOW_PLOTS", True):
        plt.show()
    plt.close(fig)


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training / Validation Loss",
    out_name: str | None = None,
    lr_history: list[float] | None = None,
):
    epochs = np.arange(1, len(train_losses) + 1)
    use_lr = lr_history is not None and len(lr_history) == len(train_losses)
    if use_lr:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        ax_loss, ax_log, ax_lr = axes[0], axes[1], axes[2]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax_loss, ax_log = axes[0], axes[1]
        ax_lr = None

    ax_loss.plot(epochs, train_losses, label="Train", color="#1f77b4", lw=1.8)
    ax_loss.plot(epochs, val_losses, label="Validation", color="#ff7f0e", lw=1.8)
    ax_loss.set_ylabel("MSE (scaled)")
    ax_loss.set_title(title)
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    ax_log.plot(epochs, train_losses, label="Train", color="#1f77b4", lw=1.5, alpha=0.85)
    ax_log.plot(epochs, val_losses, label="Validation", color="#ff7f0e", lw=1.5, alpha=0.85)
    ax_log.set_yscale("log")
    ax_log.set_ylabel("MSE (log)")
    ax_log.legend(loc="upper right")
    ax_log.grid(True, alpha=0.3, which="both")

    if ax_lr is not None:
        ax_lr.plot(epochs, lr_history, color="#2ca02c", lw=1.5)
        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_yscale("log")
        ax_lr.set_xlabel("Epoch")
        ax_lr.grid(True, alpha=0.3)
    else:
        ax_log.set_xlabel("Epoch")

    plt.tight_layout()
    _finalize(fig, out_name)


def plot_test_pred(preds: np.ndarray, trues: np.ndarray, title: str, out_name: str | None = None):
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    if preds.ndim == 1:
        preds, trues = preds.ravel(), trues.ravel()
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(trues, label="True", alpha=0.85, lw=1.2)
        ax.plot(preds, label="Predicted", alpha=0.85, lw=1.2)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Test sample index")
        ax.set_ylabel("Carbon (original scale)")
        ax.grid(True, alpha=0.3)
    else:
        h = preds.shape[1]
        fig, axes = plt.subplots(h, 1, figsize=(12, 2.3 * h), sharex=True, squeeze=False)
        for i in range(h):
            ax = axes[i, 0]
            ax.plot(trues[:, i], label="True", alpha=0.85, lw=1.2)
            ax.plot(preds[:, i], label="Predicted", alpha=0.85, lw=1.2)
            ax.set_ylabel(f"D+{i + 1}")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right")
        axes[-1, 0].set_xlabel("Test sample index")
        fig.suptitle(title, y=1.002)

    plt.tight_layout()
    _finalize(fig, out_name)


def plot_scatter_true_vs_pred(
    trues: np.ndarray,
    preds: np.ndarray,
    title: str = "True vs Predicted",
    out_name: str | None = None,
    horizon_idx: int | None = None,
):
    trues = np.asarray(trues, dtype=float).reshape(-1)
    preds = np.asarray(preds, dtype=float)
    if preds.ndim > 1:
        if horizon_idx is not None:
            preds = preds[:, horizon_idx].ravel()
            sub = f" (horizon D+{horizon_idx + 1})"
        else:
            preds = preds.reshape(-1)
            sub = " (all horizons)"
    else:
        preds = preds.ravel()
        sub = ""

    lim_min = min(trues.min(), preds.min())
    lim_max = max(trues.max(), preds.max())
    pad = (lim_max - lim_min) * 0.05 + 1e-9

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(trues, preds, alpha=0.45, s=18, edgecolors="none", c="#1f77b4")
    ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], "k--", lw=1.2, label="Ideal (y=x)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title + sub)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _finalize(fig, out_name)


def plot_residuals(
    trues: np.ndarray,
    preds: np.ndarray,
    title: str = "Residuals (Pred − True)",
    out_name: str | None = None,
):
    trues = np.asarray(trues, dtype=float)
    preds = np.asarray(preds, dtype=float)
    res = (preds - trues).ravel()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.axhline(0.0, color="k", lw=1)
    ax1.scatter(np.arange(len(res)), res, alpha=0.45, s=14, c="#9467bd")
    ax1.set_xlabel("Test sample index")
    ax1.set_ylabel("Residual")
    ax1.set_title(title + " — along test order")
    ax1.grid(True, alpha=0.3)

    ax2.hist(res, bins=40, color="#8c564b", alpha=0.85, edgecolor="white")
    ax2.axvline(0.0, color="k", lw=1.2)
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _finalize(fig, out_name)


def plot_horizon_metrics_bars(
    per_horizon: list[tuple[float, float]],
    title: str = "Per-horizon Test Metrics",
    out_name: str | None = None,
):
    if not per_horizon:
        return
    h = len(per_horizon)
    labels = [f"D+{i}" for i in range(1, h + 1)]
    rmses = [p[0] for p in per_horizon]
    r2s = [p[1] for p in per_horizon]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(h)
    w = 0.6
    ax1.bar(x, rmses, w, color="#d62728", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE by horizon")
    ax1.grid(True, alpha=0.3, axis="y")

    colors = ["#2ca02c" if r >= 0 else "#ff9896" for r in r2s]
    ax2.bar(x, r2s, w, color=colors, alpha=0.85)
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("R²")
    ax2.set_title("R² by horizon")
    pad = max(0.08, (max(r2s) - min(r2s)) * 0.15 + 1e-9)
    lo, hi = min(r2s) - pad, max(r2s) + pad
    ax2.set_ylim(lo, hi)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title)
    plt.tight_layout()
    _finalize(fig, out_name)


def plot_test_suite(
    preds_u: np.ndarray,
    trues_u: np.ndarray,
    ph: list[tuple[float, float]],
    model_prefix: str,
):
    plot_test_pred(
        preds_u,
        trues_u,
        title=f"{model_prefix}: True vs Predicted (test)",
        out_name=f"{model_prefix}_test_series.png",
    )
    plot_scatter_true_vs_pred(
        trues_u,
        preds_u,
        title=f"{model_prefix}: Scatter",
        out_name=f"{model_prefix}_scatter.png",
    )
    plot_residuals(
        trues_u,
        preds_u,
        title=f"{model_prefix}: Residuals",
        out_name=f"{model_prefix}_residuals.png",
    )
    if len(ph) > 1:
        plot_horizon_metrics_bars(
            ph,
            title=f"{model_prefix}: Metrics by horizon",
            out_name=f"{model_prefix}_horizon_metrics.png",
        )
