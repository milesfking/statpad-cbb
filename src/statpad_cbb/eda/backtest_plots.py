from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from statpad_cbb.config import load_settings
from statpad_cbb.ingest.storage import SeasonPaths

logger = logging.getLogger(__name__)


def plot_backtest(
    season: int,
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, str]:
    settings = load_settings(data_dir=data_dir)
    paths = SeasonPaths(settings.data_dir, season)
    if not paths.backtest_metrics.exists():
        raise FileNotFoundError(f"Missing {paths.backtest_metrics}; run backtest first")

    report = json.loads(paths.backtest_metrics.read_text())
    out_dir = output_dir or (paths.eda_dir / "backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    cuts = [c for c in report.get("cuts", []) if not c.get("skipped")]
    if not cuts:
        return written

    dates = [c["cut_date"] for c in cuts]
    rmse_rapm = [c.get("team", {}).get("margin_rmse_rapm") for c in cuts]
    rmse_eff = [c.get("team", {}).get("margin_rmse_eff") for c in cuts]
    rmse_blend = [c.get("team", {}).get("margin_rmse_blend") for c in cuts]
    acc_rapm = [c.get("team", {}).get("win_accuracy_rapm") for c in cuts]
    acc_eff = [c.get("team", {}).get("win_accuracy_eff") for c in cuts]
    acc_blend = [c.get("team", {}).get("win_accuracy_blend") for c in cuts]
    stab = [c.get("player_stability_vs_final", {}).get("corr_net") for c in cuts]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].plot(dates, rmse_rapm, marker="o", label="RAPM team ratings", color="#1f4e79")
    axes[0].plot(dates, rmse_eff, marker="o", label="Train efficiency", color="#c45c26")
    if any(v is not None for v in rmse_blend):
        axes[0].plot(dates, rmse_blend, marker="o", label="RAPM/eff blend", color="#2a9d8f")
    axes[0].set_title("Future game margin RMSE")
    axes[0].set_ylabel("Points")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()

    axes[1].plot(dates, acc_rapm, marker="o", label="RAPM", color="#1f4e79")
    axes[1].plot(dates, acc_eff, marker="o", label="Efficiency", color="#c45c26")
    if any(v is not None for v in acc_blend):
        axes[1].plot(dates, acc_blend, marker="o", label="Blend", color="#2a9d8f")
    axes[1].axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_title("Future home-win accuracy")
    axes[1].set_ylim(0.4, 0.9)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()

    axes[2].plot(dates, stab, marker="o", color="#1f4e79")
    axes[2].set_title("Player net stability vs final")
    axes[2].set_ylabel("Correlation")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle(f"{season} temporal backtest")
    fig.tight_layout()
    path = out_dir / "temporal_summary.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    written["temporal_summary"] = str(path)
    logger.info("wrote %s", path)

    if paths.backtest_game_predictions.exists():
        preds = pd.read_parquet(paths.backtest_game_predictions)
        if not preds.empty:
            # Use the latest cut for a scatter of predicted vs actual margins.
            latest = sorted(preds["cut_date"].unique())[-1]
            subset = preds.loc[preds["cut_date"] == latest]
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            ax.scatter(subset["pred_margin_rapm"], subset["actual_margin"], s=16, alpha=0.45, c="#1f4e79")
            lim = float(
                max(
                    abs(subset["pred_margin_rapm"]).max(),
                    abs(subset["actual_margin"]).max(),
                    1.0,
                )
            )
            ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="#999999", linewidth=1)
            ax.axhline(0, color="#cccccc", linewidth=0.8)
            ax.axvline(0, color="#cccccc", linewidth=0.8)
            ax.set_xlabel("Predicted margin (RAPM)")
            ax.set_ylabel("Actual margin")
            ax.set_title(f"Game margins after {latest}")
            path = out_dir / "margin_scatter_latest_cut.png"
            fig.tight_layout()
            fig.savefig(path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            written["margin_scatter_latest_cut"] = str(path)
            logger.info("wrote %s", path)

    return written
