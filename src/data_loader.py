"""数据读取与时序划分：先按时间顺序切分 train/val/test，min-max 只在训练段拟合，
再由各段在自身时间范围内构造滑动窗口样本组（见 src/env.py）。

prepare_folds 给出 walk-forward 多折（expanding window），用于规避单折评估
被某一特定工况（如长期停机期）主导的问题；K=1 即单折（test 取序列末尾）。
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["load_series", "prepare_folds", "describe_series"]


def load_series(path, column: Optional[str] = None) -> Tuple[np.ndarray, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p.resolve()}")
    df = pd.read_excel(p) if p.suffix.lower() in ('.xlsx', '.xls', '.xlsm') else pd.read_csv(p)
    if column is None:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 1:
            raise ValueError(f"{p.name} 中未找到数值列")
        column = num.columns[0]
    if column not in df.columns:
        raise KeyError(f"列 '{column}' 不存在；可用列：{list(df.columns)}")
    v = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError(f"{p.name}::{column} 含 {int((~np.isfinite(v)).sum())} 个缺失/非数值，请先在源数据中处理")
    if (v < 0).any():
        raise ValueError(f"{p.name}::{column} 含负值")
    return v, str(column)


def _build_segment(s: np.ndarray, tr_end: int, va_end: int, te_end: int) -> Dict:
    """train=[0,tr_end) val=[tr_end,va_end) test=[va_end,te_end)；min-max 只用训练段拟合。"""
    n = len(s)
    if not (0 < tr_end < va_end < te_end <= n):
        raise ValueError(f"时间段非法：train=[0,{tr_end}) val=[{tr_end},{va_end}) "
                         f"test=[{va_end},{te_end})，N={n}")
    lo, hi = float(s[:tr_end].min()), float(s[:tr_end].max())
    hi = hi if hi - lo > 1e-12 else lo + 1.0
    f = lambda x: (x - lo) / (hi - lo)
    return dict(train=f(s[:tr_end]), val=f(s[tr_end:va_end]), test=f(s[va_end:te_end]),
                raw_train=s[:tr_end], raw_val=s[tr_end:va_end], raw_test=s[va_end:te_end],
                lo=lo, hi=hi, zero_level=(0.0 - lo) / (hi - lo),
                n_train=tr_end, n_val=va_end - tr_end, n_test=te_end - va_end,
                offset_val=tr_end, offset_test=va_end, span=(0, tr_end, va_end, te_end))


def prepare_folds(series, n_folds: int = 1, train_ratio: float = 0.80,
                  val_ratio: float = 0.10) -> List[Dict]:
    """Walk-forward 折划分（expanding window）。

    K=1 时 test 取序列末尾 1-train_ratio-val_ratio 的部分。
    K>1：每折测试段长度 T=(1-train_ratio-val_ratio)×N，第 i 折测试段为
        [N-(K-i)·T, N-(K-i-1)·T)，验证段紧随其前，训练段为 [0, 验证段起点)。
    每折的 min-max 只由该折训练段拟合，且训练数据在时间上严格早于该折测试数据。
    """
    s = np.asarray(series, dtype=np.float64)
    n, K = len(s), int(n_folds)
    if K < 1:
        raise ValueError(f"n_folds 需 ≥ 1，实得 {n_folds}")
    tr, va = float(train_ratio), float(val_ratio)
    test_ratio = 1.0 - tr - va
    if min(tr, va, test_ratio) <= 0:
        raise ValueError(f"train_ratio/val_ratio 非法：train={tr}, val={va}, test={test_ratio}")
    # 用整数差分定义段长，避免 1-0.8-0.1=0.09999999999999998 这类浮点误差导致切分点偏移。
    tr_end_base, va_end_base = int(n * tr), int(n * (tr + va))
    val_size, test_size = va_end_base - tr_end_base, n - va_end_base
    if min(test_size, val_size) < 1:
        raise ValueError(f"数据过短：N={n} 无法构造 test_size={test_size}, val_size={val_size}")
    out = []
    for i in range(K):
        te_end = n - (K - 1 - i) * test_size
        va_end = te_end - test_size
        tr_end = va_end - val_size
        if tr_end <= 0:
            raise ValueError(f"第 {i + 1}/{K} 折训练段为空（N={n} 偏短）")
        out.append(_build_segment(s, tr_end, va_end, te_end))
    return out


def _max_zero_run(x) -> int:
    best = cur = 0
    for t in x:
        cur = cur + 1 if t == 0 else 0
        best = max(best, cur)
    return best


def describe_series(seg: Dict) -> Dict:
    out = {}
    for name in ('train', 'val', 'test'):
        r = seg[f'raw_{name}']
        on = (r > 0).astype(int)
        out[name] = dict(zero=float(np.mean(r == 0)), max_zero_run=_max_zero_run(r),
                         switch_per_100=float(np.abs(np.diff(on)).sum() / max(len(r) - 1, 1) * 100),
                         max_norm=float(np.max(seg[name])))
    dz = np.abs(np.diff(seg['train']))
    out['train_dy_q'] = np.quantile(dz, [0.5, 0.9, 0.95, 0.99])
    return out
