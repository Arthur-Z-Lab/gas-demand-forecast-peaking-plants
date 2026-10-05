"""论文绘图统一风格（本项目唯一绘图入口）。

本模块即绘图约定的唯一来源，所有绘图脚本都应：

    from src.plot_style import sci_rc, save_fig, model_color, panel_label, LIGHT_CMAP
    sci_rc()
    ...
    save_fig(fig, "output/figures/main/main_MAE")

约定要点：Times New Roman、tick 朝内且首尾有刻度、无网格、图例无边框、
**面板标签写 "(a) 子图标题" 并置于横轴标题下方**、折线与热图都要显示 mean ± SD、
需要时用空心点/粗体显示 Diebold–Mariano 显著性、只输出 SVG + PNG(600 dpi)、图不放标题。
"""
import colorsys
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

__all__ = ["sci_rc", "save_fig", "deepen", "model_color", "panel_label", "panel_caption",
           "palette_color", "LIGHT_CMAP", "LIGHT_DIVERGING", "LIGHT_SEQUENTIAL",
           "CAL_DIVERGING", "CAL_SEQUENTIAL",
           "DEGRADE_SEQUENTIAL",
           "MODEL_COLORS", "SAVEFIG_KWARGS",
           "PALETTE_SUPERVISED", "PALETTE_RL", "PALETTE_PROBABILISTIC",
           "LW_MAIN", "LW_AUX", "LW_THIN", "AXIS_LW", "MARKER_SIZE", "CAPSIZE", "BAND_ALPHA"]

# 多模型配色：与基线节专用色板同一套色相序列（第 2 个对照永远是深蓝、第 3 个永远是绿……），
# 保证同一语义在不同小节拿到一致的颜色；本文模型仍用橙红强调（见 MAIN_COLOR）。
# 多模型配色：低饱和「中性族」——对照模型统一退到冷色/中性色，本文模型的橙红强调色
# 因此始终是图里最醒目的一条（Nature 风格：一个中性族 + 一个强调族，而不是七彩并置）。
MODEL_COLORS = ["#2E5A88", "#3F7F7A", "#6A6FA6", "#A2648C",
                "#6E8B3D", "#8A6E4B", "#5C6B73", "#9AA5AD"]
MAIN_COLOR = "#EC5E1F"          # 本文模型（CBR-TD3）的强调色

# ---------------------------------------------------------------- 线宽/标记
# 全项目统一：主曲线 > 对照曲线 > 区间/参考线；标记与误差棒帽同尺度
LW_MAIN = 1.6
LW_AUX = 1.1
LW_THIN = 0.9
AXIS_LW = 0.8
MARKER_SIZE = 3.4
CAPSIZE = 2.2
BAND_ALPHA = 0.18

# ---------------------------------------------------------------- 基线下装配色
# 原则：本文模型固定用 MAIN_COLOR（亮橙红）强调；基线一律用低饱和深色、色相拉开、
# 白底与灰度打印均可区分，且不出现与强调色接近的橙黄系。
PALETTE_SUPERVISED = ["#2E5A88", "#3F7F7A", "#6A6FA6", "#A2648C",
                      "#6E8B3D", "#8A6E4B", "#5C6B73", "#9AA5AD"]
PALETTE_RL = ["#2E5A88", "#3F7F7A", "#6A6FA6", "#A2648C", "#6E8B3D", "#5C6B73"]
PALETTE_PROBABILISTIC = ["#2E5A88", "#3F7F7A", "#6A6FA6", "#A2648C",
                         "#6E8B3D", "#8A6E4B", "#5C6B73", "#9AA5AD"]


def palette_color(name: str, palette, is_main: bool = False, index: int = 0) -> str:
    """基线对比配色：本文模型用强调色，其余按给定色板顺序取色。"""
    if is_main:
        return MAIN_COLOR
    return list(palette)[int(index) % len(list(palette))]

SAVEFIG_KWARGS = {"bbox_inches": "tight", "pad_inches": 0.03, "facecolor": "white"}

# 指标热图：浅色系（越浅越好），避免深底白字
LIGHT_CMAP = LinearSegmentedColormap.from_list(
    "light_metrics", ["#FFFFFF", "#EAF3FA", "#CFE4F5", "#A9CDEA", "#7CB1DA", "#4C8FBF"], N=256)

# 发散色板：用于「偏离参考值」的热图（覆盖率相对标称、消融/扰动的相对变化）。
# 两端亮度仍 >0.4，保证单元格文字可以统一用黑色（黑字对 0.4 亮度背景的对比度约 8:1）。
LIGHT_DIVERGING = LinearSegmentedColormap.from_list(
    "light_diverging", ["#3B75B5", "#A8C9E5", "#FFFFFF", "#F0BFAE", "#C0553C"], N=256)

# 单向色板（较深的蓝，与 LIGHT_DIVERGING 的蓝端同色相）：用于区间宽度、Winkler 这类
# 「越小越好」的量，保证与发散图同属一套视觉语言。
LIGHT_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "light_sequential", ["#FFFFFF", "#E3EEF8", "#C2DBEE", "#96BEE0", "#659DC9"], N=256)

# ---- 概率预测热图专用（6.1.3）------------------------------------------------
# 校准覆盖率的发散色板：以标称 95% 为中点，欠覆盖走蓝、过覆盖走红、正中近白。
# 低饱和、两端亮度 > 0.45，保证单元格文字统一用黑色仍可读（对比度 ≥ 7:1）。
CAL_DIVERGING = LinearSegmentedColormap.from_list(
    "cal_diverging", ["#1F5C99", "#7FA8D4", "#C9D9EA", "#F4F3F1", "#EBC0B6", "#C2665B", "#A63A32"],
    N=256)

# 概率预测的「成本型」单向色板（PINAW / Winkler，越小越好）：青绿单向渐变，
# 与发散图同为一套视觉语言，但色相与覆盖率面板明确区分，避免三格看起来是同一张图。
CAL_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "cal_sequential", ["#F7FBFA", "#D6E9E6", "#A9D2CD", "#74B4AE", "#418F8C", "#1F6B6E"], N=256)

# 退化热图（6.3.2 扰动鲁棒性）：全部为「相对干净输入变差」的单向量，用暖色单向渐变
# 比发散色更贴合语义；两端亮度 > 0.45，保证黑色单元格文字可读。
DEGRADE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "degrade_sequential",
    ["#FFFBF7", "#FDE9D9", "#F8CDA9", "#F0A877", "#E07C46", "#BE4E2A", "#93321C"], N=256)


def sci_rc(bold: bool = True, size: float = 9.0):
    """统一 rcParams。bold=True 时 tick / 轴标题 / 图例 / 标注加粗（论文要求）。"""
    plt.rcParams.update({
        "font.family": "serif",
        # Linux 上依次回退；Windows 环境首选 Times New Roman
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                       "DejaVu Serif", "Noto Serif CJK SC"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": float(size),
        "axes.titlesize": float(size),
        "axes.labelsize": float(size),
        "legend.fontsize": 8.5,        # 与子图轴标题（8.5）一致，保证图例可读
        "xtick.labelsize": float(size),
        "ytick.labelsize": float(size),
        "axes.grid": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.facecolor": "white",
    })
    if bold:
        plt.rcParams.update({
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "legend.fontsize": 8.5,
        })


def save_fig(fig, stem, dpi: int = 600, svg: bool = True, png: bool = True):
    """保存为 SVG + PNG(600 dpi)（本项目规范：不输出 PDF）。返回已写出的路径列表。"""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    if svg:
        p = stem.with_suffix(".svg")
        fig.savefig(p, **SAVEFIG_KWARGS)
        out.append(p)
    if png:
        p = stem.with_suffix(".png")
        fig.savefig(p, dpi=dpi, **SAVEFIG_KWARGS)
        out.append(p)
    plt.close(fig)
    return out


def deepen(hex_color: str, factor: float = 0.78) -> str:
    """白底图配色加深（HSL 亮度 × factor），避免浅色发飘。"""
    r, g, b = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l2 = max(0.18, l * factor)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s)
    return f"#{int(round(r2 * 255)):02X}{int(round(g2 * 255)):02X}{int(round(b2 * 255)):02X}"


def model_color(name: str, is_main: bool = False, index: int = 0) -> str:
    """按模型名取色：本文模型用强调色，其余按索引取对比色。"""
    if is_main or str(name).lower() in ("ubc-td3", "ours", "proposed", "本文模型"):
        return MAIN_COLOR
    return MODEL_COLORS[int(index) % len(MODEL_COLORS)]


def panel_label(ax, text: str, x: float = -0.10, y: float = 1.02, fontsize: float = None):
    """（旧写法，保留兼容）面板标签置于子图内部左上角。新图请用 ``panel_caption``。"""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold", fontsize=fontsize or plt.rcParams["font.size"])


def panel_caption(ax, letter: str, title: str = "", linespacing: float = 1.8,
                  fontsize: float = None, xlabel: str = None):
    """面板标签新规范：**(a) 子图标题** 置于**横轴标题正下方**（间距一行，不过远）。

    实现方式是把横轴标题写成两行——第一行是坐标轴标题，第二行是面板标签，
    因此标签永远贴在横轴标题下方，且 tight_layout 会正确预留空间。

    用法::

        ax.set_xlabel("Forecast horizon (day)")
        panel_caption(ax, "a", "MAE")        # -> 显示为 "Forecast horizon (day)\\n(a) MAE"
    """
    lab = (xlabel if xlabel is not None else ax.get_xlabel()).strip()
    txt = f"({letter}) {title}".strip()
    ax.set_xlabel(f"{lab}\n{txt}" if lab else txt, fontsize=fontsize,
                  linespacing=float(linespacing))


def style_axes(ax, xticks=None, yticks=None, hide_xlabel: bool = False):
    """统一坐标轴：tick 朝内、首尾真实刻度、可选隐藏横轴标签（多面板矩阵用）。"""
    ax.tick_params(axis="both", direction="in")
    ax.grid(False)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if hide_xlabel:
        ax.tick_params(labelbottom=False)
