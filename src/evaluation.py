"""评估。

点预测 = actor 后验均值 rollout（纯策略，无任何线性兜底）；
预测区间 = Bayesian 策略后验采样（每组后验权重做一次 H 步递归 rollout，取分位数），
           即框架自带的不确定性表达，不做额外的共形/异方差校准。
"""
from typing import Dict, Tuple

import numpy as np

from src.metrics import (crps_ensemble, interval_metrics, nll_gaussian, point_metrics,
                         q_bias, ramp_violation, reliability_curve, reserve_shortfall)

__all__ = ["val_score", "mc_rollouts", "fit_conformal", "evaluate"]

# 校准验证用的置信水平网格（可靠性图 / 多水平覆盖率）
PI_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


def _apply_blend(ro: Dict, blend_w, blend_prior, gated: bool = True) -> Dict:
    """可选：把策略点预测与线性先验按 ŷ = w·ŷ_policy + (1−w)·ŷ_prior 混合（原地写入 ro['P']）。

    gated=True 时只对模型判定为**开机**的步混合，停机步保留 hurdle 门控输出。
    仅在 --use_prior 1 时启用（消融"线性先验兜底对精度是否有用"）。
    """
    if blend_w is None or blend_prior is None:
        return ro
    w = float(blend_w)
    P = np.asarray(ro['P'], dtype=np.float64)
    R = np.asarray(blend_prior, dtype=np.float64)[:, np.asarray(ro['starts'], dtype=np.int64)].T
    mixed = w * P + (1.0 - w) * R
    if gated and 'OFF' in ro:
        mixed = np.where(np.asarray(ro['OFF'], bool), P, mixed)
    ro['P'] = mixed
    return ro


def fit_conformal(env, agent, n_samples: int, level: float,
                  blend_w=None, blend_prior=None, levels=None):
    """按 horizon 校准绝对残差半宽（split conformal）。

    返回 (q_h, q_map)：
      q_h   —— 主置信水平 level 的半宽（用于主区间）；
      q_map —— {置信水平: 半宽} 的网格（用于可靠性曲线/多水平覆盖率），levels 为空时只算主水平。
    """
    _ = n_samples
    ro = env.rollout(agent.act_batch, env.starts)
    ro = _apply_blend(ro, blend_w, blend_prior)
    score = np.abs(ro['Y'] - ro['P'])
    n = len(env.starts)

    def _quant(lv: float) -> np.ndarray:
        q = min(1.0, np.ceil((n + 1) * float(lv)) / n)
        return np.quantile(score, q, axis=0, method='higher')

    q_h = _quant(level)
    q_map = {float(lv): _quant(lv) for lv in (levels or [])}
    return q_h, q_map


def val_score(env, agent, mode: str = 'mae') -> float:
    """验证集分数（早停与检查点选择依据）。只使用验证集标签，不使用测试集信息。

    mode='mae'       ：整体 MAE（与论文主指标一致，默认）
    mode='balanced'  ：按 ON/OFF/切换三类等权组合，避免停机样本被自然频率淹没
                       （更重视运行状态，但会牺牲整体误差）
    """

    ro = env.rollout(
        agent.act_batch,
        env.starts
    )

    P = np.asarray(ro['P'], dtype=float)
    Y = np.asarray(ro['Y'], dtype=float)

    err = np.abs(P - Y)
    overall = float(err.mean())
    if mode == 'mae':
        return overall

    true_off = (
        Y <= env.z0 + 1e-9
    )

    true_on = ~true_off

    # 前一日真实状态
    prev_y = np.concatenate(
        [
            ro['anchors'].reshape(-1, 1),
            Y[:, :-1]
        ],
        axis=1
    )

    prev_off = (
        prev_y <= env.z0 + 1e-9
    )

    switch = (
        true_off != prev_off
    )

    mae_on = (
        float(err[true_on].mean())
        if true_on.any()
        else overall
    )

    mae_off = (
        float(err[true_off].mean())
        if true_off.any()
        else overall
    )

    mae_switch = (
        float(err[switch].mean())
        if switch.any()
        else overall
    )

    # 不按验证集自然工况比例加权
    score = (
        0.50 * mae_on
        + 0.25 * mae_off
        + 0.25 * mae_switch
    )

    return float(score)


def mc_rollouts(env, agent, starts, n_samples: int) -> np.ndarray:
    out = np.zeros((n_samples, len(starts), env.H))
    for i in range(n_samples):
        agent.resample_policy()
        out[i] = env.rollout(lambda s: agent.act_batch(s, mode='fixed'), starts)['P']
    return out


def evaluate(env, agent, cfg, *, scale: float, offset: float, max_lag: int,
             n_samples: int, level: float, blend_w=None, blend_prior=None,
             conformal_q=None, pi_method: str = 'bayes',
             ramp_limit: float = None, conformal_qmap: Dict = None) -> Tuple[Dict, Dict]:
    starts = env.starts
    ro = env.rollout(agent.act_batch, starts)
    ro = _apply_blend(ro, blend_w, blend_prior)
    m = point_metrics(ro, cfg=cfg, scale=scale, offset=offset, max_lag=max_lag)

    B, H = ro['P'].shape
    S = ro['S'][:, :H]
    q1, qm = agent.q_values(S.reshape(B * H, -1), ro['A'].reshape(B * H, 2))
    m.update(q_bias(q1.reshape(B, H), qm.reshape(B, H), ro['R'], agent.gamma))

    # 预测区间：Bayesian 策略后验采样 → 逐步分位数
    Ys = mc_rollouts(env, agent, starts, n_samples)
    a = (1 - level) / 2
    lo_b, hi_b = np.quantile(Ys, a, 0), np.quantile(Ys, 1 - a, 0)
    # ---- 区间与校准验证（审稿人核心要求）----
    # 三套口径：
    #   PICP/PINAW       主口径（--pi_method 指定）
    #   Bayes_* / Cal_*  贝叶斯后验采样 / 共形校准（同时输出，便于两套对照）
    #   BayesPICP{l}     贝叶斯后验采样在置信水平 l 下的实测覆盖率（可靠性曲线，用来看"贝叶斯不确定性是否校准"）
    #   CalPICP{l}       共形校准在同一网格上的实测覆盖率（校准后应≈l）
    iv_bayes = interval_metrics(ro['Y'], lo_b, hi_b, level, scale, "")
    m.update({f"Bayes_{k}": v for k, v in iv_bayes.items()})
    ro.update(lo_bayes=lo_b, hi_bayes=hi_b, levels_bayes_lo={}, levels_bayes_hi={})

    lo_by_l, hi_by_l = {}, {}
    for lv in PI_LEVELS:
        q = (1.0 - lv) / 2.0
        lo_by_l[lv] = np.quantile(Ys, q, axis=0)
        hi_by_l[lv] = np.quantile(Ys, 1 - q, axis=0)
        m[f'BayesPICP{int(round(lv * 100))}'] = float(np.mean((ro['Y'] >= lo_by_l[lv]) & (ro['Y'] <= hi_by_l[lv])))
    lv_list, emp, cerr = reliability_curve(ro['Y'], lo_by_l, hi_by_l)
    m['BayesCalErrMeanAbs'] = float(np.mean(np.abs(cerr)))
    ro.update(levels_bayes_lo=lo_by_l, levels_bayes_hi=hi_by_l)

    # 共形校准：验证段残差在每个置信水平下的分位数（验证集拟合，测试集评估）
    primary, lo_p, hi_p = iv_bayes, lo_b, hi_b
    if conformal_q is not None:
        qh = np.asarray(conformal_q, float).reshape(1, H)
        lo_c = np.maximum(ro['P'] - qh, cfg.zero_level)
        hi_c = ro['P'] + qh
        iv_cal = interval_metrics(ro['Y'], lo_c, hi_c, level, scale, "")
        m.update({f"Cal_{k}": v for k, v in iv_cal.items()})
        ro.update(lo_cal=lo_c, hi_cal=hi_c)
        if str(pi_method) == 'conformal':
            primary, lo_p, hi_p = iv_cal, lo_c, hi_c
    if conformal_qmap:
        for lv, qarr in conformal_qmap.items():
            qq = np.asarray(qarr, float).reshape(1, H)
            l_l = np.maximum(ro['P'] - qq, cfg.zero_level)
            h_h = ro['P'] + qq
            m[f'CalPICP{int(round(lv * 100))}'] = float(np.mean((ro['Y'] >= l_l) & (ro['Y'] <= h_h)))
            m[f'CalPINAW{int(round(lv * 100))}'] = float((h_h - l_l).mean() / (float(np.ptp(ro['Y'])) or 1.0))
            ro.setdefault('levels_cal_lo', {})[lv] = l_l
            ro.setdefault('levels_cal_hi', {})[lv] = h_h
        if ro.get('levels_cal_lo'):
            lv2, emp2, cerr2 = reliability_curve(ro['Y'], ro['levels_cal_lo'], ro['levels_cal_hi'])
            m['CalCalErrMeanAbs'] = float(np.mean(np.abs(cerr2)))
    m.update(primary)
    ro.update(lo_primary=lo_p, hi_primary=hi_p)

    # ---- 概率预测质量：CRPS / NLL / 后验离散度 ----
    m['CRPS'] = float(crps_ensemble(Ys, ro['Y']).mean() * scale)
    mu_s, sd_s = Ys.mean(0), Ys.std(0)
    m['MC_sigma_mean'] = float(sd_s.mean() * scale)
    # ---- 运行安全指标（供燃料调度 / 备用容量与爬坡规划解读，论文报 1–2 个即可）----
    y_o, p_o = ro['Y'] * scale + offset, ro['P'] * scale + offset
    anchors_o = np.asarray(ro['anchors'], float)[:, None] * scale + offset
    prev_true = np.concatenate([anchors_o, y_o[:, :-1]], 1)
    if ramp_limit is None:
        ramp_limit = float(np.quantile(np.abs(np.diff(np.asarray(ro['hist'], float))), 0.95)) * scale
    m['RampLimit'] = float(ramp_limit)
    m['ReserveMargin'] = 0.05 * scale
    m.update(reserve_shortfall(y_o.ravel(), p_o.ravel(), scale=scale, margin_frac=0.05))
    m.update(ramp_violation(y_o.ravel(), prev_true.ravel(), float(ramp_limit)))
    # 上界越限率：按预测区间的上界安排备用时的越限频率（对应备用容量规划）
    m['UpperExceedRate'] = float(np.mean(ro['Y'] > hi_p))
    m['LowerExceedRate'] = float(np.mean(ro['Y'] < lo_p))
    m['NLL_bayes'] = float(nll_gaussian(ro['Y'], mu_s, np.maximum(sd_s, 1e-8)))
    # 校准后（共形半径换算成等效高斯 σ）的 NLL —— 论文主张"校准良好的概率预测"时报这个
    if conformal_q is not None:
        from scipy.stats import norm as _norm
        z = float(_norm.ppf(0.5 * (1.0 + level)))
        sig_c = np.asarray(conformal_q, float).reshape(1, H) / max(z, 1e-6)
        sig_full = np.broadcast_to(sig_c, ro['Y'].shape)
        m['NLL'] = float(nll_gaussian(ro['Y'], ro['P'], np.maximum(sig_full, 1e-8)))
        m['MC_to_cal_sigma_ratio'] = float(sd_s.mean() / max(sig_c.mean(), 1e-8))
    else:
        m['NLL'] = m['NLL_bayes']

    ro.pop('S')
    ro.update(lo_bayes=lo_b, hi_bayes=hi_b)
    return m, ro
