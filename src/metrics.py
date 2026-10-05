from typing import Dict

import numpy as np

from model.paper.reward import lcar

__all__ = ["accuracy", "lag_diagnostics", "change_metrics", "onoff_metrics",
           "reward_stats", "q_bias", "interval_metrics",
           "crps_ensemble", "point_metrics",
           "ple", "mase", "dtw_distance", "dm_test",
           "nll_gaussian", "reliability_curve", "reserve_shortfall", "ramp_violation",
           "block_bootstrap_test"]

_EPS = 1e-12


def accuracy(y, p) -> Dict[str, float]:
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    e = p - y
    ok = y.std() > _EPS and p.std() > _EPS
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                R2=float(1 - np.sum(e ** 2) / (np.sum((y - y.mean()) ** 2) + _EPS)),
                CORR=float(np.corrcoef(y, p)[0, 1]) if ok else float("nan"),
                Bias=float(np.mean(e)))


def lag_diagnostics(y, p, max_lag: int) -> Dict[str, float]:
    """corr_lag{k} = corr(预测序列, 真值序列向后错位 k 步)；best_lag>0 仅作滞后诊断信号，不能单独证明复制。"""
    out, best, bc = {}, 0, -np.inf
    for k in range(max_lag + 1):
        a, b = p[k:], y[:len(y) - k]
        if len(a) > 3 and a.std() > _EPS and b.std() > _EPS:
            c = float(np.corrcoef(a, b)[0, 1])
            out[f"corr_lag{k}"] = c
            if c > bc:
                best, bc = k, c
    out["best_lag"] = float(best)
    return out


def change_metrics(Y, P, anchors, dir_eps: float, y_peak: float, scale: float) -> Dict[str, float]:
    a = np.asarray(anchors, float)[:, None]
    dy = np.diff(np.concatenate([a, Y], 1), axis=1)
    dp = np.diff(np.concatenate([a, P], 1), axis=1)
    m = np.abs(dy) > dir_eps
    big = np.abs(dy) >= np.quantile(np.abs(dy), 0.9)
    peak = Y >= y_peak
    E = P - Y
    return dict(DirAcc=float(np.mean(np.sign(dp[m]) == np.sign(dy[m]))) if m.any() else float("nan"),
                MAE_delta=float(np.mean(np.abs(dp - dy)) * scale),
                MAE_bigchange=float(np.mean(np.abs(E[big])) * scale),
                RMSE_peak=float(np.sqrt(np.mean(E[peak] ** 2)) * scale) if peak.any() else float("nan"))


def onoff_metrics(Y, P, anchors, z0: float, on_eps: float, scale: float,
                  pred_off=None, p_off=None) -> Dict[str, float]:
    """启停总体指标 + 四类真实状态转移（ON→ON/ON→OFF/OFF→OFF/OFF→ON）。"""
    Y, P = np.asarray(Y, float), np.asarray(P, float)
    t_on = Y > z0 + 1e-9
    if pred_off is None:
        p_on = P > z0 + on_eps
    else:
        p_on = ~np.asarray(pred_off, bool)
    prev = np.concatenate([np.asarray(anchors, float)[:, None], Y[:, :-1]], 1) > z0 + 1e-9
    sw = t_on != prev
    ae = np.abs(P - Y) * scale
    correct = (t_on == p_on)

    out = dict(OnOffAcc=float(correct.mean()),
               ShutdownRate=float((~t_on).mean()),
               OffRecall=float(np.mean(~p_on[~t_on])) if (~t_on).any() else float("nan"),
               OnRecall=float(np.mean(p_on[t_on])) if t_on.any() else float("nan"),
               PredShutdownRate=float((~p_on).mean()),
               MAE_on=float(ae[t_on].mean()) if t_on.any() else float("nan"),
               MAE_off=float(ae[~t_on].mean()) if (~t_on).any() else float("nan"),
               MAE_switch=float(ae[sw].mean()) if sw.any() else float("nan"),
               Switch_frac=float(sw.mean()))

    # 停机段误差对总误差的贡献占比（只用本模型误差，不涉及任何对照）
    tot_ae = float(ae.mean())
    out['ErrShare_off'] = (float((~t_on).sum() * ae[~t_on].mean() / (ae.size * max(tot_ae, _EPS)))
                           if (~t_on).any() else float("nan"))

    trans = {
        'ON_ON': prev & t_on,
        'ON_OFF': prev & (~t_on),
        'OFF_OFF': (~prev) & (~t_on),
        'OFF_ON': (~prev) & t_on,
    }
    for name, mask in trans.items():
        out[f'Acc_{name}'] = float(correct[mask].mean()) if mask.any() else float('nan')
        out[f'N_{name}'] = float(mask.sum())

    if p_off is not None:
        po = np.clip(np.asarray(p_off, float), 1e-6, 1 - 1e-6)
        target_off = (~t_on).astype(float)
        out['OffBrier'] = float(np.mean((po - target_off) ** 2))
        out['OffLogLoss'] = float(np.mean(-(target_off * np.log(po) + (1 - target_off) * np.log(1 - po))))
    return out


def reward_stats(Y, P, anchors, cfg):
    R, parts = lcar(P, Y, anchors, cfg)
    return R, dict(Reward_per_step=float(R.mean()), DirMissRate=float(parts['dir_miss'].mean()),
                   StateMissRate=float(parts['state_miss'].mean()))


def q_bias(Q1, Qmin, R, gamma: float) -> Dict[str, float]:
    """G_k = Σ_{j≥k} γ^{j−k} r_j（H 步终止，无截断）；偏差 = Q(s_k, a_k) − G_k。"""
    G = np.zeros_like(R)
    acc = np.zeros(len(R))
    for k in range(R.shape[1] - 1, -1, -1):
        acc = R[:, k] + gamma * acc
        G[:, k] = acc
    return dict(Return_mean=float(G.mean()), QBias_q1=float(np.mean(Q1 - G)),
                QBias_min=float(np.mean(Qmin - G)),
                QBias_rel=float(np.mean(Qmin - G) / (np.mean(np.abs(G)) + _EPS)),
                QOver_frac=float(np.mean(Qmin > G)))


def crps_ensemble(samples, y) -> np.ndarray:
    """样本 CRPS：E|X−y| − ½E|X−X'|，samples 形状 (S, …)。"""
    S = samples.shape[0]
    t1 = np.mean(np.abs(samples - y[None]), 0)
    xs = np.sort(samples, 0)
    w = (2 * np.arange(1, S + 1) - S - 1).reshape(-1, *([1] * (samples.ndim - 1)))
    t2 = np.sum(w * xs, 0) / (S * (S - 1))
    return t1 - t2


def interval_metrics(Y, lo, hi, level: float, scale: float, prefix: str) -> Dict[str, float]:
    """区间质量指标（论文口径）：

        PICP  = (1/N) Σ I(L_t ≤ y_t ≤ U_t)                        —— 区间覆盖率（标定质量）
        PINAW = (1/(N(y_max − y_min))) Σ (U_t − L_t)               —— 归一化平均宽度（锐度）
        Winkler = 区间得分（Gneiting & Raftery, 2007），作为补充指标
    """
    a = 1 - level
    cover = (Y >= lo) & (Y <= hi)
    width = hi - lo
    rng = float(Y.max() - Y.min()) or 1.0
    wink = width + (2 / a) * (lo - Y) * (Y < lo) + (2 / a) * (Y - hi) * (Y > hi)
    out = {f"{prefix}PICP": float(cover.mean()), f"{prefix}PINAW": float(width.mean() / rng),
           f"{prefix}Winkler": float(wink.mean() * scale)}
    H = Y.shape[1]
    for h in (0, H - 1):
        out[f"{prefix}PICP_h{h + 1}"] = float(cover[:, h].mean())
    return out


# ---------------------------------------------------------------------------
# 峰值误差 / 尺度无关误差 / 序列形状距离（论文主表指标）
# ---------------------------------------------------------------------------
def ple(y, p) -> float:
    """峰值负荷误差 PLE(%)：|max(y) − max(ŷ)| / max(y) × 100。"""
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    m = float(np.max(y))
    return float(abs(m - float(np.max(p))) / m * 100.0) if m > _EPS else float('nan')


def mase(y, p, y_hist, season: int = 7) -> float:
    """MASE：MAE 除以历史序列的朴素季节性误差尺度（尺度无关，便于跨数据集比较）。"""
    y, p, h = np.asarray(y, float).ravel(), np.asarray(p, float).ravel(), np.asarray(y_hist, float).ravel()
    if len(h) <= season:
        return float('nan')
    scale = float(np.mean(np.abs(h[season:] - h[:-season])))
    return float(np.mean(np.abs(p - y)) / scale) if scale > _EPS else float('nan')


def dtw_distance(y, p, band: int = None) -> float:
    """DTW 距离（欧氏代价，Sakoe-Chiba 带宽可选）。y/p 为等长一维序列。"""
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    n, m = len(y), len(p)
    if n == 0 or m == 0:
        return float('nan')
    band = int(band) if band else max(n, m)
    inf = float('inf')
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j0, j1 = max(1, i - band), min(m, i + band)
        for j in range(j0, j1 + 1):
            cost = abs(y[i - 1] - p[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def dm_test(e1, e2, h: int = 1) -> Dict[str, float]:
    """Diebold–Mariano 检验（双侧，Newey–West HAC 方差）。

    e1/e2：两个模型在同一批样本上的**误差序列**（如绝对误差或平方误差）。
    返回 dict(dm, p, mean_diff)；p<0.05 表示两模型预测精度差异显著。
    """
    e1, e2 = np.asarray(e1, float).ravel(), np.asarray(e2, float).ravel()
    d = e1 - e2
    n = len(d)
    if n < 8 or not np.isfinite(d).all():
        return dict(dm=float('nan'), p=float('nan'), mean_diff=float(np.nanmean(d)) if n else float('nan'))
    dbar = float(d.mean())
    d0 = d - dbar
    gamma0 = float(np.dot(d0, d0) / n)
    var = gamma0
    for lag in range(1, max(1, int(h))):
        g = float(np.dot(d0[lag:], d0[:-lag]) / n)
        var += 2.0 * (1.0 - lag / (h if h > 1 else 1)) * g
    if var <= 0:
        return dict(dm=float('nan'), p=1.0, mean_diff=dbar)
    dm = dbar / np.sqrt(var / n)
    from scipy.stats import norm as _norm
    return dict(dm=float(dm), p=float(2 * (1 - _norm.cdf(abs(dm)))), mean_diff=dbar)


def block_bootstrap_test(e1, e2, block: int = 10, n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    """块自助法（moving block bootstrap）比较两个模型的误差序列均值差。

    适用于存在自相关的预测误差：按块重采样误差差序列，给出均值差的置信区间与 p 值。
    """
    e1, e2 = np.asarray(e1, float).ravel(), np.asarray(e2, float).ravel()
    d = e1 - e2
    n = len(d)
    if n < block * 2:
        return dict(mean_diff=float(np.mean(d)) if n else float("nan"),
                    lo=float("nan"), hi=float("nan"), p=float("nan"))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    obs = float(d.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    # 双侧 p 值：中心化后 |均值| 超过观测的块自助分布频率
    centered = means - obs
    p = float(np.mean(np.abs(centered) >= abs(obs)))
    return dict(mean_diff=obs, lo=float(lo), hi=float(hi), p=p)


def nll_gaussian(y, mu, sigma) -> float:
    """高斯负对数似然 NLL：用预测分布的均值/标准差衡量概率预测质量（越小越好）。"""
    y, mu, sigma = (np.asarray(a, float).ravel() for a in (y, mu, sigma))
    s = np.clip(sigma, 1e-8, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + (y - mu) ** 2 / (2 * s ** 2)))


def reliability_curve(Y, lo_by_level: Dict[float, np.ndarray], hi_by_level: Dict[float, np.ndarray]):
    """可靠性（校准）曲线：名义覆盖率 → 实际覆盖率；返回 (levels, empirical, cal_err)。"""
    Y = np.asarray(Y, float)
    levels = sorted(lo_by_level.keys())
    emp = [float(np.mean((Y >= lo_by_level[l]) & (Y <= hi_by_level[l]))) for l in levels]
    return levels, emp, [float(e - l) for l, e in zip(levels, emp)]


def reserve_shortfall(y, p, scale: float = 1.0, margin_frac: float = 0.05) -> Dict[str, float]:
    """运行安全指标 1：备用容量不足（供气/调峰备用规划）。

    预测偏低超过 margin（默认 5% 量程）的日数占比 + 平均缺口（原始单位），
    直接对应"按预测值安排调峰备用时的失负荷风险"。
    """
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    margin = float(margin_frac) * float(scale)
    gap = y - p - margin
    return dict(ReserveShortfallRate=float(np.mean(gap > 0)),
                ReserveShortfallMW=float(np.mean(np.maximum(gap, 0.0))))


def ramp_violation(y, prev_y, limit: float) -> Dict[str, float]:
    """运行安全指标 2：爬坡越限（供气/机组爬坡能力规划）。

    实际日间变化 |Δy| 超过爬坡限值 limit 的比例（限值取训练段 |Δy| 的高分位数），
    对应"按预测安排爬坡能力时的越限风险"。
    """
    y, prev = np.asarray(y, float).ravel(), np.asarray(prev_y, float).ravel()
    d = np.abs(y - prev)
    return dict(RampViolationRate=float(np.mean(d > float(limit))),
                RampExcessMW=float(np.mean(np.maximum(d - float(limit), 0.0))))


def point_metrics(ro: Dict, *, cfg, scale: float, offset: float, max_lag: int,
                  ) -> Dict[str, float]:
    Y, P, anc = ro['Y'], ro['P'], ro['anchors']
    B, H = Y.shape
    m = {f"{k}_norm": v for k, v in accuracy(Y, P).items()}
    m.update(accuracy(Y * scale + offset, P * scale + offset))
    # 论文主表指标：峰值负荷误差 PLE(%)、尺度无关 MASE、序列形状 DTW
    y_o, p_o = Y * scale + offset, P * scale + offset
    m['PLE'] = ple(y_o.ravel(), p_o.ravel())
    y_hist = np.asarray(ro.get('hist', []), float).ravel()
    m['MASE'] = mase(Y.ravel() * scale, P.ravel() * scale, y_hist * scale) if y_hist.size else float('nan')
    m['DTW'] = dtw_distance(y_o[:, 0], p_o[:, 0])          # 首步轨迹（长度 = 起报日数）
    m['DTW_hH'] = dtw_distance(y_o[:, -1], p_o[:, -1])     # 末步轨迹
    for h in range(H):                                     # 逐步长 PLE / DTW
        m[f'PLE_h{h + 1}'] = ple(y_o[:, h], p_o[:, h])
        m[f'DTW_h{h + 1}'] = dtw_distance(y_o[:, h], p_o[:, h])
    for h in range(H):
        acc_h = accuracy(Y[:, h], P[:, h])
        m[f"MAE_h{h + 1}"] = acc_h['MAE'] * scale
        m[f"R2_h{h + 1}"] = acc_h['R2']
    m.update(lag_diagnostics(Y[:, 0], P[:, 0], max_lag))
    m[f"best_lag_h{H}"] = lag_diagnostics(Y[:, -1], P[:, -1], max_lag)['best_lag']
    m.update(change_metrics(Y, P, anc, cfg.dir_eps, cfg.y_peak, scale))
    m.update(onoff_metrics(Y, P, anc, cfg.zero_level, cfg.on_eps, scale,
                           pred_off=ro.get('OFF'), p_off=ro.get('OFFP')))
    R, rs = reward_stats(Y, P, anc, cfg)
    m.update(rs)
    m.update(n_origins=float(B), horizon=float(H))
    ro['R'] = R
    return m
