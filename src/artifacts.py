import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["check_resume", "append_row", "summarize", "write_json", "write_predictions"]

try:                       # POSIX：用 flock 串行化同一 trials 文件的“读—改—写”
    import fcntl
except ImportError:        # Windows：无 fcntl，退化为无锁（与原行为一致）
    fcntl = None


@contextmanager
def _trials_lock(path: Path):
    """同一 trials_*.csv 的跨进程互斥锁。

    并行跑同一 (H, fold) 的不同 seed 时，多个进程会同时调用 append_row；
    append_row 是“先读整表、再回写”，无锁会让后写者静默覆盖先写者刚追加的行。
    锁文件与数据文件同目录（<name>.lock），Windows 上无 fcntl 时退化为无锁。
    """
    if fcntl is None:
        yield
        return
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """先写同目录临时文件再 os.replace：读方要么看到旧文件，要么看到完整新文件。"""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def check_resume(path: Path, config_json: str) -> set:
    with _trials_lock(path):
        if not path.exists() or path.stat().st_size == 0:
            return set()
        df = pd.read_csv(path)
        old = {json.dumps(json.loads(c), sort_keys=True) for c in df['config'].dropna().unique()}
        if old != {json.dumps(json.loads(config_json), sort_keys=True)}:
            raise SystemExit(f"[已中止] {path.name} 记录的配置与本次不同；请删除该文件或更换 --outdir。")
        return set(df['seed'].astype(int))


def append_row(path: Path, row: dict) -> pd.DataFrame:
    new = pd.DataFrame([row])
    with _trials_lock(path):
        df = pd.concat([pd.read_csv(path), new], ignore_index=True) if path.exists() else new
        df = df.drop_duplicates(subset=['seed'], keep='last')
        _atomic_write_csv(df, path)
    return df


def summarize(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).drop(columns=['seed'], errors='ignore')
    s = pd.concat([num.mean().rename('mean'), num.std(ddof=1).rename('std')], axis=1)
    s['n_seeds'] = len(df)
    # seed 仅用于量化优化随机性；不把多个 seed 当作独立测试样本统计显著率。
    s.to_csv(out_path)
    return s


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    os.replace(tmp, path)


def write_predictions(path: Path, ro: dict, scale: float, offset: float, seg_offset: int) -> None:
    B, H = ro['P'].shape
    f = lambda x: np.asarray(x).ravel() * scale + offset
    tgt = (seg_offset + ro['starts'])[:, None] + np.arange(H)[None, :]
    df = pd.DataFrame({'origin_idx': np.repeat(seg_offset + ro['starts'], H),
                       'h': np.tile(np.arange(1, H + 1), B), 'target_idx': tgt.ravel(),
                       'y_true': f(ro['Y']), 'y_pred': f(ro['P']),
        'y_on_head': f(ro['P_ON']), 'p_off': np.asarray(ro['OFFP']).ravel(),
        'pred_off': np.asarray(ro['OFF']).ravel().astype(int),
        'survival_feature': np.asarray(ro['SURV']).ravel()})
    if 'lo_primary' in ro:          # 主口径区间（--pi_method 指定）
        df['lo_pred_interval'] = f(ro['lo_primary'])
        df['hi_pred_interval'] = f(ro['hi_primary'])
    if 'lo_bayes' in ro:
        df['lo_bayes'] = f(ro['lo_bayes'])
        df['hi_bayes'] = f(ro['hi_bayes'])
    if 'lo_cal' in ro:              # 共形校准区间（两套口径对照）
        df['lo_cal'] = f(ro['lo_cal'])
        df['hi_cal'] = f(ro['hi_cal'])
    df.to_csv(path, index=False)
