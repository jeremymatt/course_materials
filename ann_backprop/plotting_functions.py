"""Plotting helpers for main.py. Kept separate so the network code stays readable."""

import matplotlib
matplotlib.use("Agg")          # draw straight to image files; no window needed
import matplotlib.pyplot as plt
import numpy as np

TRAIN_COLOR = "#D77A49"
VAL_COLOR = "#5B7F9C"
NET_COLOR = "#1E2B24"
SINE_COLOR = "#9CA3AF"


def _add_warning(fig, warning):
    """Print a warning (e.g. for the step activation) across the top of a figure."""
    if warning:
        fig.text(0.5, 0.995, warning, ha="center", va="top", color="#C0392B",
                 fontsize=10, fontweight="bold")


def plot_losses(train_losses, val_losses, path, warning=None):
    """Training and validation loss after every epoch (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, color=TRAIN_COLOR, label="training loss")
    ax.plot(val_losses, color=VAL_COLOR, label="validation loss")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    ax.grid(alpha=0.3)
    ax.legend()
    _add_warning(fig, warning)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if warning else 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_snapshots(snapshots, x_plot, x_train, y_train, x_val, y_val, path, warning=None, title=""):
    """Four panels: the network's output at the start, 1/3, 2/3 and end of training.

    snapshots: dict {epoch: network output at x_plot}
    The shaded areas are outside the range of the training data.
    """
    epochs = sorted(snapshots)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    lo, hi = x_train.min(), x_train.max()

    for ax, epoch in zip(axes.ravel(), epochs):
        ax.axvspan(x_plot.min(), lo, color="0.92", zorder=0)
        ax.axvspan(hi, x_plot.max(), color="0.92", zorder=0)
        ax.plot(x_plot, np.sin(x_plot), color=SINE_COLOR, lw=2, label="sin(x)")
        ax.plot(x_plot, snapshots[epoch], ".", color=NET_COLOR, ms=3, label="network output")
        ax.plot(x_train, y_train, "o", color=TRAIN_COLOR, ms=7, label="training points")
        ax.plot(x_val, y_val, "s", color=VAL_COLOR, ms=7, label="validation points")
        ax.set_title("Initial random weights" if epoch == 0 else f"After epoch {epoch}")
        ax.set_ylim(-2.5, 2.5)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="lower left", fontsize=9)
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("y")
    axes[0, 0].text(0.99, 0.97, "shaded = outside training data", transform=axes[0, 0].transAxes,
                    ha="right", va="top", fontsize=9, color="0.4")

    fig.suptitle(title, y=0.95 if warning else 0.98)
    _add_warning(fig, warning)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if warning else 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
