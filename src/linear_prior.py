"""线性先验多步预测：只使用起报时刻可见的历史窗口，无未来信息。

用途：作为**门控混合**的另一支——在模型判定为开机的步上，
最终预测 = w·策略预测 + (1−w)·线性先验预测，权重 w 在验证段上选（见 select_blend_weight）。

防泄漏：
  * 验证/测试段：只用训练段拟合的模型；
  * 训练段本身：walk-forward 分块交叉拟合（第 i 块只用其之前的块），
    避免“在同一段上既拟合又评估”带来的乐观偏差。
"""
from typing import Dict

import numpy as np

__all__ = ["build_features", "segment_priors", "select_blend_weight"]


def build_features(y: np.ndarray, idx: np.ndarray, n_lag: int = 14) -> np.ndarray:
    """起报时刻 t 的特征：滞后 1..n_lag + 滚动统计（全部来自 t-1 及更早）。"""
    y = np.asarray(y, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    cols = [y[idx - j] for j in range(1, int(n_lag) + 1)]
    cols += [np.stack([y[i - 7:i].mean() for i in idx]),
             np.stack([y[i - 14:i].mean() for i in idx]),
             np.stack([y[i - 14:i].min() for i in idx]),
             np.stack([y[i - 14:i].max() for i in idx]),
             np.stack([np.mean(y[i - 7:i] <= 0.0) for i in idx]),
             (y[idx - 1] <= 0.0).astype(np.float64)]
    return np.column_stack(cols)


def _ridge_fit_predict(Xtr, ytr, Xte, alpha: float) -> np.ndarray:
    from sklearn.linear_model import Ridge
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = Ridge(alpha=float(alpha)).fit((Xtr - mu) / sd, ytr)
    return m.predict((Xte - mu) / sd)


def _prior_block(fit_y: np.ndarray, fit_idx: np.ndarray, pred_y: np.ndarray,
                 starts: np.ndarray, horizon: int, alpha: float, n_lag: int,
                 fallback: np.ndarray) -> np.ndarray:
    """用 fit_y[fit_idx] 训练，在 pred_y 的 starts 起报点上给出 (H, len(starts)) 先验。

    fit_y 与 pred_y 必须是不同段（训练段 vs 验证/测试段）**分别传入**：
    早先版本把 pred_y 当训练数组用，等于让线性先验在测试段自拟合（信息泄漏）。
    """
    H = int(horizon)
    fit_y = np.asarray(fit_y, dtype=np.float64)
    pred_y = np.asarray(pred_y, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    out = np.tile(np.asarray(fallback, float).reshape(1, -1), (H, 1))
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    fit_idx = fit_idx[(fit_idx >= n_lag + 1) & (fit_idx <= len(fit_y) - H)]
    if len(fit_idx) < 30 or len(starts) == 0:
        return out
    Xtr, Xte = build_features(fit_y, fit_idx, n_lag), build_features(pred_y, starts, n_lag)
    for k in range(H):
        out[k] = _ridge_fit_predict(Xtr, fit_y[fit_idx + k], Xte, alpha)
    hi = float(np.percentile(fit_y[fit_idx], 99.5)) * 1.5 + 1e-6
    return np.clip(out, 0.0, hi)


def segment_priors(seg: Dict, seq_len: int, horizon: int, alpha: float = 1.0,
                   n_lag: int = 14, n_blocks: int = 5) -> Dict[str, np.ndarray]:
    """给 train/val/test 三段分别生成线性先验矩阵（每段形状 (H, len(段))）。

    约定 R[k, t] 为 t 起报、未来第 k+1 步的预测；特征不足的起报点回退为持久化（t-1 的值）。
    """
    L, H = int(seq_len), int(horizon)
    out: Dict[str, np.ndarray] = {}
    lo = max(L, int(n_lag) + 1)
    tr = np.asarray(seg["train"], dtype=np.float64)
    R_tr = np.full((H, len(tr)), np.nan)
    edges = np.linspace(lo, len(tr), int(n_blocks) + 1).astype(int)
    for b in range(int(n_blocks)):
        s0, s1 = edges[b], edges[b + 1]
        if s1 <= s0:
            continue
        starts = np.arange(s0, s1)
        fit_idx = np.arange(lo, s0) if b > 0 else np.array([], dtype=np.int64)
        R_tr[:, s0:s1] = _prior_block(tr, fit_idx, tr, starts, H, alpha, n_lag,
                                      fallback=tr[np.clip(starts - 1, 0, None)])
    bad = ~np.isfinite(R_tr)
    if bad.any():
        cols = np.where(bad)[1]
        R_tr[bad] = tr[np.clip(cols - 1, 0, None)]
    out["train"] = R_tr

    fit_idx = np.arange(lo, len(tr))
    for name in ("val", "test"):
        y = np.asarray(seg[name], dtype=np.float64)
        starts = np.arange(lo, len(y))
        R = np.full((H, len(y)), np.nan)
        if len(starts):
            R[:, lo:] = _prior_block(tr, fit_idx, y, starts, H, alpha, n_lag,
                                     fallback=y[np.clip(starts - 1, 0, None)])
        bad = ~np.isfinite(R)
        if bad.any():
            cols = np.where(bad)[1]
            R[bad] = y[np.clip(cols - 1, 0, None)]
        out[name] = R
    return out


def select_blend_weight(env, agent, prior: np.ndarray, metric: str = 'mse',
                        grid=None) -> Dict:
    """在**验证段**上选择"策略预测 vs 线性先验"的混合权重 w（固化流程的一步）：

        ŷ = w·ŷ_policy + (1−w)·ŷ_linear

    只在模型判定为**开机**的步上混合；停机步保留 hurdle 门控输出（避免把停机段的
    强项交给线性先验）。权重在验证段上按 MSE 选择，测试段不参与，无信息泄漏。

    返回 dict(weight, metric, score, mse, mae, table=[...])。
    """
    grid = np.round(np.arange(0.0, 1.001, 0.05), 3) if grid is None else np.asarray(grid, float)
    ro = env.rollout(agent.act_batch, env.starts)
    Y = np.asarray(ro['Y'], float)
    P = np.asarray(ro['P'], float)
    R = np.asarray(prior, float)[:, np.asarray(ro['starts'], int)].T
    off = np.asarray(ro['OFF'], bool) if 'OFF' in ro else None
    table, best = [], None
    for w in grid:
        pred = w * P + (1.0 - w) * R
        if off is not None:
            pred = np.where(off, P, pred)
        e = pred - Y
        mse, mae = float(np.mean(e ** 2)), float(np.mean(np.abs(e)))
        table.append(dict(weight=float(w), mse=mse, mae=mae))
        key = mse if metric == 'mse' else mae
        if best is None or key < best[0] - 1e-15:
            best = (key, float(w))
    row = next(r for r in table if r['weight'] == best[1])
    return dict(weight=float(best[1]), metric=str(metric), score=float(best[0]),
                mse=float(row['mse']), mae=float(row['mae']), table=table)
