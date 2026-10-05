"""训练循环（预算 = 固定 episode 数）。

每一轮：从训练段 episode 集合中无放回抽取 n_parallel 个起报日（集合抽完后重新打乱）；
冻结一组后验策略权重并加高斯动作噪声（标准 TD3 探索），并行执行 H 步"预测—写回—再预测"；
n_parallel×H 条转移写入经验回放池；随后执行 round(utd × n_parallel × H) 次 TD3 更新。
每 val_every 个 episode 在验证段按时间顺序评估一次，早停并回滚到验证最优的策略参数。
"""
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch

__all__ = ["train", "build_agent"]


def train(agent, env, *, total_episodes: int, n_parallel: int, utd: float, warmup_episodes: int,
          rng: np.random.Generator, val_fn: Optional[Callable], val_every: int,
          patience: int, min_episodes: int, log_every_sec: float) -> Dict:
    starts = env.starts
    if len(starts) < n_parallel:
        print(f"    [train] ⚠ 训练段仅 {len(starts)} 个起报日 < --n_parallel={n_parallel}，"
              f"将按实际可抽取数量并行（建议下调 --n_parallel）")
    pool = np.array([], dtype=np.int64)
    ep = upd = 0
    next_val = val_every
    best, best_ep, best_state, bad = np.inf, 0, None, 0
    curve, ep_reward = [], []
    t0 = t_last = time.time()

    def emit(v=None):
        """控制台只显示验证指标；优化器、梯度更新和训练奖励保留在日志文件中。"""
        nonlocal t_last
        if v is not None:
            nb = int(round(20 * ep / total_episodes))
            print(f"    [val] ▕{'█' * nb}{'░' * (20 - nb)}▏ episode {ep:,}/{total_episodes:,}"
                  f" │ MAE_norm {v:.5f} │ best {best:.5f} @ep{best_ep:,}", flush=True)
        t_last = time.time()

    print(f"    [train] {total_episodes:,} episode；每 {val_every:,} episode 输出一次验证 MAE")
    while ep < total_episodes:
        if len(pool) < n_parallel:
            pool = np.concatenate([pool, rng.permutation(starts)])
        idx, pool = pool[:n_parallel], pool[n_parallel:]
        n_eff = len(idx)

        if ep < warmup_episodes:
            # v3 动作为 [delta, p_off]；warmup 只随机探索 delta，状态概率先置 0，避免随机停机污染回放池。
            policy = lambda s: np.column_stack([
                rng.uniform(agent.a_low, agent.a_high, size=len(s)),
                np.zeros(len(s), dtype=float)])
        else:
            agent.begin_episodes(n_eff, ep / total_episodes)
            policy = agent.explore
        ro = env.rollout(policy, idx)
        trans, R = env.transitions(ro)
        agent.store(**trans)
        ep += n_eff
        ep_reward.append(float(R.sum(1).mean()))

        for _ in range(max(1, int(round(utd * n_eff * env.H)))):
            if agent.learn() is not None:
                upd += 1

        v = None
        if val_fn is not None and ep >= next_val:
            next_val += val_every
            v = float(val_fn(agent))
            curve.append((ep, v))
            if v < best - 1e-9:
                best, best_ep, bad, best_state = v, ep, 0, agent.actor_state()
            else:
                bad += 1
            emit(v)
            if ep >= min_episodes and bad >= patience:
                print(f"    [train] 早停 @episode {ep:,}，回滚到 episode {best_ep:,}（val MAE_norm {best:.5f}）")
                break
        elif time.time() - t_last >= log_every_sec:
            # metric-only console mode: non-validation training diagnostics are not printed
            t_last = time.time()
    if best_state is not None:
        agent.load_actor_state(best_state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # 训练过程曲线（每轮 episode 奖励、验证曲线）随 agent.logs 一起落盘；
    # 只新增日志字段，不改变训练行为。此前这两条曲线在 run.py 里取完标量后被丢弃，
    # 导致事后无法复盘"奖励是否在涨"。
    agent.logs['episode_reward'] = ep_reward
    agent.logs['val_curve'] = curve
    # 若 total_episodes < val_every，验证从未触发、best 仍为 inf：
    # 记作 NaN 而不是 inf，避免污染 summary 的均值/标准差
    best_val = float(best) if np.isfinite(best) else float('nan')
    return dict(val_curve=curve, episode_reward=ep_reward, best_episode=best_ep,
                best_val=best_val, episodes_run=ep, updates=upd, transitions=ep * env.H)


def build_agent(
    env,
    model_cls,
    model_kwargs: Dict,
    *,
    a_low: float,
    a_high: float
):
    """
    所有转移先验只从训练段计算。

    start_prior:
        P(OFF_t | ON_{t-1})

    stay_prior:
        P(OFF_t | OFF_{t-1})
    """

    off = (
        env.y <= env.z0 + 1e-9
    ).astype(np.float64)

    prev = off[:-1]
    nxt = off[1:]

    # ON -> OFF
    mask_on = prev < 0.5

    if mask_on.sum() > 0:
        start_prior = float(
            nxt[mask_on].mean()
        )
    else:
        start_prior = 0.03

    # OFF -> OFF
    mask_off = prev >= 0.5

    if mask_off.sum() > 0:
        stay_prior = float(
            nxt[mask_off].mean()
        )
    else:
        stay_prior = 0.70

    start_prior = float(
        np.clip(
            start_prior,
            1e-3,
            1.0 - 1e-3
        )
    )

    stay_prior = float(
        np.clip(
            stay_prior,
            1e-3,
            1.0 - 1e-3
        )
    )

    return model_cls(
        n_features=env.state_dim,
        seq_len=env.L,
        n_aux=env.aux_dim,
        horizon=env.H,
        a_low=a_low,
        a_high=a_high,
        phys_low=env.z0,

        off_start_prior=start_prior,
        off_stay_prior=stay_prior,

        off_threshold=env.off_threshold,
        gate_temp=env.gate_temp,

        lam_delta=env.cfg.lam_delta,
        point_scale=env.cfg.point_scale,

        **model_kwargs
    )
