"""结果图：run.py 每个种子训练结束后自动调用，图片保存在 <outdir>/figures/。"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plot_style import panel_label, save_fig, sci_rc, style_axes

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
sci_rc()

__all__ = ["plot_all"]

C_PRED, C_BAND = "#C0392B", "#F0B27A"


def _save(fig, path):
    """统一出口：SVG + PNG(600 dpi)，bbox tight、白底（规范 §2）。"""
    try:
        fig.tight_layout()
        save_fig(fig, path.with_suffix(""))     # path 传入 .png 名义路径，实际写 svg+png
    except Exception as e:                      # 绘图失败不应中断主流程
        print(f"    [警告] 保存图片 {path.name} 失败（已跳过）：{e}")
        plt.close(fig)


def plot_forecast(df, out, stem, h, H):
    d = df[df.h == h]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(d.target_idx, d.lo_bayes, d.hi_bayes, color=C_BAND, alpha=0.45, lw=0,
                    label="Bayesian PI")
    ax.plot(d.target_idx, d.y_true, color="k", lw=1.1, label="Observed")
    ax.plot(d.target_idx, d.y_pred, color=C_PRED, lw=1.0, label=f"Forecast (day {h} of {H})")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Gas load (m$^3$)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    style_axes(ax)
    _save(fig, out / f"{stem}_forecast_h{h}.png")


def plot_horizon(df, out, stem):
    g = df.assign(m=(df.y_pred - df.y_true).abs()).groupby("h")[["m"]].mean()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(g.index, g["m"], marker="o", color=C_PRED, label="Proposed")
    ax.set_xlabel("Forecast horizon (day)")
    ax.set_ylabel("MAE (m$^3$)")
    ax.legend(frameon=False)
    style_axes(ax, xticks=list(g.index))
    _save(fig, out / f"{stem}_mae_by_horizon.png")


def plot_scatter(df, out, stem, H):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharex=True, sharey=True)
    # 防御性下界：NaN 会污染 max() 使 lim 变成 NaN，零跨度（全零真值）会让坐标轴退化
    yv = np.asarray(df.y_true, float)
    pv = np.asarray(df.y_pred, float)
    yv, pv = yv[np.isfinite(yv)], pv[np.isfinite(pv)]
    hi = max(float(yv.max()) if yv.size else 1.0, float(pv.max()) if pv.size else 1.0)
    lim = [0.0, hi * 1.05 if hi > 0 else 1.0]
    for ax, h in zip(axes, (1, H)):
        d = df[(df.h == h) & np.isfinite(df.y_true) & np.isfinite(df.y_pred)]
        if len(d):
            ax.scatter(d.y_true, d.y_pred, s=8, alpha=0.6, color=C_PRED)
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        panel_label(ax, f"({chr(97 + list((1, H)).index(h))}) Day {h}", x=-0.12, y=1.02)
        ax.set_xlabel("Observed")
        style_axes(ax)
    axes[0].set_ylabel("Forecast")
    _save(fig, out / f"{stem}_scatter.png")


def plot_all(pred_csv, log, logs, out_dir, stem, H):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_csv)
    for h in sorted({1, H}):
        plot_forecast(df, out, stem, h, H)
    plot_horizon(df, out, stem)
    plot_scatter(df, out, stem, H)
    # 只保留评估结果图；训练/梯度/Q 等内部诊断曲线不在此函数输出。
    return out
