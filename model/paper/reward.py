"""负荷变化感知的多步预测奖励（Load-Change-Aware Reward, LCAR）。

第 k 步（k = 1..H）：
    r_k = -ω_k · [ w_k·ℓ(e_k) + λ_Δ·|Δŷ_k − Δy_k|
                 + λ_dir·1(|Δy_k|>δ ∧ sgn Δŷ_k ≠ sgn Δy_k) + λ_s·1(S(ŷ_k) ≠ S(y_k)) ]
    e_k = ŷ_k − y_k,  Δŷ_k = ŷ_k − ŷ_{k−1},  Δy_k = y_k − y_{k−1},  ŷ_0 = y_0 = 起报前一日实测值
    w_k = 1 + β_Δ·min(|Δy_k|/Δ_ref, 3) + β_p·σ(κ(y_k − y_peak))
    S(v) = 1(v > z0 + ε_on)，z0 为停机水平

三项对应论文的奖励设计：**预测误差**（ℓ=e²，平方误差，与 R²/RMSE 目标一致）、
**负荷变化一致性**（Δ 惩罚 + 方向惩罚）、**运行状态转移**（启停状态错判惩罚）。
另有 λ_g·growth 抑制递归多步误差逐步放大（多步序列决策特有）。
"""
from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["RewardConfig", "make_reward_config", "lcar"]


@dataclass
class RewardConfig:
    horizon: int = 7
    # 奖励类型：lcar=本文提出的负荷变化感知奖励；其余为替换式对照（整体替换，不叠加 LCAR 其余项）
    #   quantile = 分位数损失（τ=0.5，等价 MAE 口径）
    #   pinball  = Pinball 损失（默认 τ=0.9，偏重高负荷侧）
    #   wmae     = 加权平均绝对误差（权重 = LCAR 的变化/峰段权重 W）
    #   pwmse    = 峰值加权均方误差（只保留峰段加权）
    reward_type: str = 'lcar'
    pinball_tau: float = 0.9
    point_scale: float = 6.0
    beta_change: float = 1.0
    change_ref: float = 0.10
    beta_peak: float = 0.5
    y_peak: float = 0.70
    kappa: float = 12.0
    lam_delta: float = 0.10
    lam_dir: float = 0.01
    dir_eps: float = 0.02
    lam_state: float = 0.03
    lam_growth: float = 0.15
    growth_margin: float = 0.02
    # ---- 可选奖励项（--reward_extras 1 时启用，用于消融其精度贡献）----
    reward_extras: bool = False
    eta_h: float = 0.0
    quality_weight: float = 0.05
    quality_ref: float = 0.08
    quality_temp: float = 0.04
    sparse_weight: float = 0.10
    sparse_ref: float = 0.10
    sparse_temp: float = 0.04
    zero_level: float = 0.0
    on_eps: float = 0.01
    clip_reward: Optional[float] = 20.0

    def __post_init__(self):
        assert self.horizon >= 1, "horizon 需 ≥ 1"
        assert self.reward_type in ('lcar', 'quantile', 'pinball', 'wmae', 'pwmse'), \
            f"reward_type 不支持：{self.reward_type}"
        assert 0.0 < self.pinball_tau < 1.0, "pinball_tau 必须在 (0,1) 内"
        assert self.change_ref > 0 and self.dir_eps >= 0
        assert self.point_scale > 0
        assert self.quality_temp > 0 and self.sparse_temp > 0

    def to_dict(self) -> Dict:
        return asdict(self)


def make_reward_config(horizon: int, overrides: Optional[Dict] = None) -> RewardConfig:
    params = {}
    if overrides:
        valid = {f.name for f in fields(RewardConfig)} - {"horizon"}
        bad = sorted(set(overrides) - valid)
        if bad:
            raise KeyError(f"奖励参数不存在：{bad}")
        params = {k: v for k, v in overrides.items() if v is not None}
    return RewardConfig(horizon=int(horizon), **params)


def lcar(P, Y, anchors, cfg: RewardConfig, off_prob=None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """统一 dense+sparse 奖励。

    Dense 主项保持 6×weighted-MSE；quality 项把“很好/一般/很差”拉开；
    growth 只小权重抑制递归误差继续放大；最后一步加入 episode-MAE 稀疏奖励。
    不使用任何基础控制器或 residual controller。
    """
    P, Y = np.asarray(P, float), np.asarray(Y, float)
    a = np.asarray(anchors, float).reshape(-1, 1)
    H = Y.shape[1]
    prev_true = np.concatenate([a, Y[:, :-1]], 1)
    prev_hat = np.concatenate([a, P[:, :-1]], 1)
    d_true, d_hat = Y - prev_true, P - prev_hat
    W = 1.0 + cfg.beta_change * np.minimum(np.abs(d_true) / cfg.change_ref, 3.0)
    if cfg.beta_peak > 0:
        peak_excess = np.maximum((Y - cfg.y_peak) / max(1.0 - cfg.y_peak, 1e-6), 0.0)
        W = W + cfg.beta_peak * peak_excess ** 2
    if H > 1 and cfg.reward_extras:
        Om = np.broadcast_to(1.0 + cfg.eta_h * np.arange(H) / (H - 1), Y.shape)
    else:
        Om = np.broadcast_to(np.ones(1), Y.shape)

    e = P - Y
    ae = np.abs(e)
    target_off = (Y <= cfg.zero_level + 1e-9).astype(float)

    # ---- 替换式奖励（奖励函数消融：整体替换 LCAR，不叠加其余项）----
    if cfg.reward_type != 'lcar':
        if cfg.reward_type == 'wmae':                 # 加权 MAE：沿用 LCAR 的变化/峰段权重 W
            loss = cfg.point_scale * W * ae
        elif cfg.reward_type == 'pwmse':              # 峰值加权 MSE：只保留峰段加权
            if cfg.beta_peak > 0:
                peak_excess = np.maximum((Y - cfg.y_peak) / max(1.0 - cfg.y_peak, 1e-6), 0.0)
                peak_w = 1.0 + cfg.beta_peak * peak_excess ** 2
            else:
                peak_w = 1.0
            loss = cfg.point_scale * peak_w * e ** 2
        else:                                          # quantile(τ=0.5) / pinball(τ 可调)
            tau = 0.5 if cfg.reward_type == 'quantile' else float(cfg.pinball_tau)
            err = Y - P                                # 目标 − 预测
            loss = cfg.point_scale * np.where(err >= 0.0, tau * err, (tau - 1.0) * err)
        R = -loss
        if cfg.clip_reward is not None:
            c = abs(float(cfg.clip_reward))
            R = np.clip(R, -c, c)
        z = np.zeros_like(Y)
        return R, dict(W=W, Om=np.ones_like(Y), prev_true=prev_true, dir_miss=z,
                       state_miss=z.astype(bool), state_loss=z, growth=z)

    point = cfg.point_scale * W * e ** 2
    delta = cfg.lam_delta * (d_hat - d_true) ** 2
    dir_miss = ((np.abs(d_true) > cfg.dir_eps) & (np.sign(d_hat) != np.sign(d_true))).astype(float)

    if off_prob is None:
        state_prob = (P <= cfg.zero_level + cfg.on_eps).astype(float)
    else:
        state_prob = np.clip(np.asarray(off_prob, float), 0.0, 1.0)
    state_loss = cfg.lam_state * (state_prob - target_off) ** 2

    prev_ae = np.concatenate([ae[:, :1], ae[:, :-1]], axis=1)
    growth_raw = np.maximum(ae - prev_ae - cfg.growth_margin, 0.0)
    growth_raw[:, 0] = 0.0
    growth = cfg.lam_growth * growth_raw ** 2

    R = Om * (-point - delta - cfg.lam_dir * dir_miss - state_loss - growth)
    quality = sparse = np.zeros_like(R)
    if cfg.reward_extras:
        # 可选：① 分段质量奖励（把"很好/一般/很差"拉开）② episode 末步 MAE 稀疏奖励
        quality = cfg.quality_weight * np.tanh((cfg.quality_ref - ae) / cfg.quality_temp)
        mae_ep = ae.mean(axis=1)
        sparse = cfg.sparse_weight * np.tanh((cfg.sparse_ref - mae_ep) / cfg.sparse_temp)
        R = R + Om * quality
        R[:, -1] += sparse
    if cfg.clip_reward is not None:
        c = abs(float(cfg.clip_reward))
        R = np.clip(R, -c, c)
    return R, dict(W=W, Om=np.array(Om), prev_true=prev_true, dir_miss=dir_miss,
                   state_miss=(state_prob >= 0.5) != (target_off >= 0.5),
                   state_loss=state_loss, growth=growth, quality=quality, sparse=sparse)
