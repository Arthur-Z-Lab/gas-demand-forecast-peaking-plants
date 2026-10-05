"""历史数据驱动的多步预测环境（v3：显式启停状态 + 条件正负荷）。

一个 episode = 一个起报日 t0 的未来 H 日预测：
  s_k = [窗口 L 日, k/H, 连续停机计数, 经验停机持续概率]
  策略输出 u_k = [a_k, p_off,k]：
      a_k      : 正负荷分支的残差动作
      p_off,k  : 未来第 k 步处于停机状态的概率
  正负荷分支：ŷ_on = clamp(base + a_k, z0, p_max)
  最终点预测：默认采用 hurdle hard gate，p_off >= threshold 时 ŷ=z0，否则 ŷ=ŷ_on。

关键区别：经验 P(停|已停 c 天) 只作为状态特征，不再直接乘到预测值上；
连续停机计数会随模型自己的启停预测逐步更新，因此允许 ON→OFF 与 OFF→ON 转换。
"""
from typing import Callable, Dict

import numpy as np

from model.paper.reward import lcar

__all__ = ["ForecastEnv"]


class ForecastEnv:

    def __init__(
            self,
            series,
            seq_len: int,
            reward_cfg,
            min_start: int = 0,
            stop_eps: float = 1e-9,
            p_max: float = None,
            stop_survival: bool = False,
            surv_table=None,
            off_threshold: float = 0.5,
            gate_temp: float = 0.08,
            gate_mode: str = 'hurdle',
            truth=None,
            delay: int = 0,
    ):
        # series = 模型**可见**的输入序列（鲁棒性实验里可被缺失/噪声/波动污染）
        # truth  = 用于评估的真值序列（默认与输入相同；缺失/噪声扰动时保持原始观测）
        self.y = np.asarray(series, dtype=np.float64).ravel()
        self.truth = self.y if truth is None else np.asarray(truth, dtype=np.float64).ravel()
        self.delay = int(delay)          # 通信延迟：可见历史滞后 delay 天
        self.L = int(seq_len)
        self.H = int(reward_cfg.horizon)
        self.cfg = reward_cfg
        self.z0 = float(reward_cfg.zero_level)
        self.stop_eps = float(stop_eps)
        self.p_max = float(p_max) if p_max is not None else float('inf')
        self.stop_survival = bool(stop_survival)
        self.off_threshold = float(off_threshold)
        if not (0.0 < self.off_threshold < 1.0):
            raise ValueError(f"off_threshold 必须在 (0,1) 内，实得 {self.off_threshold}")
        if surv_table is not None:
            self.stop_survival = True
        first = max(self.L + self.delay, int(min_start) + self.delay)
        self.starts = np.arange(first, len(self.y) - self.H + 1, dtype=np.int64)
        if len(self.starts) == 0:
            raise ValueError(f"序列过短（{len(self.y)}），无法构造 L={self.L}, H={self.H} 的样本组")

        # 截至每日末的连续停机天数。训练段用于估计生存先验；推理时 rollout 内会按预测状态更新。
        self._run = np.zeros(len(self.y), dtype=np.int64)
        c = 0
        for t in range(len(self.y)):
            c = c + 1 if self.y[t] <= self.stop_eps else 0
            self._run[t] = c
        self.stop_scale = float(np.log1p(60.0))
        self._surv = np.asarray(surv_table, dtype=np.float64) if surv_table is not None \
            else (self._fit_survival() if self.stop_survival else None)

        # horizon progress + run-length + survival prior
        self.aux_dim = 3
        self.state_dim = self.L + self.aux_dim

        # 温度门控
        self.gate_temp = float(gate_temp)
        # 门控：'hurdle' = p_off 软门控把预测拉向停机水平（原 v8 设计）；
        #       'none'   = 点预测直接用正负荷分支（p_off 仅用于状态判定与指标）
        self.gate_mode = str(gate_mode)
        if self.gate_mode not in ('hurdle', 'none'):
            raise ValueError(f"gate_mode 只支持 hurdle/none，实得 {self.gate_mode}")

        if self.gate_temp <= 0:
            raise ValueError(
                f"gate_temp 必须 > 0，实得 {self.gate_temp}"
            )
    @property
    def n_episodes(self) -> int:
        return len(self.starts)

    def _stop_feature(self, run):
        """连续停机天数的有界归一化：log1p(n)/log1p(60)。"""
        return np.clip(np.log1p(np.asarray(run, dtype=np.float64)) / self.stop_scale, 0.0, 1.0)

    def _fit_survival(self, min_n: int = 3, max_c: int = 7) -> np.ndarray:
        """训练段经验 p(c)=P(次日仍停 | 已连续停机 c 天)，仅作为状态先验特征。

        对训练段未覆盖的长停机使用单调 logit-log 外推；它不再直接缩放最终负荷预测，
        因而即使外推存在误差，也只是一个可被网络忽略/修正的输入特征。
        """
        y, n = self.y, len(self.y)
        out = np.zeros(n + 1)
        for c in range(1, max_c + 1):
            idx = np.arange(n - 1)[self._run[:n - 1] == c]
            if len(idx) >= min_n:
                out[c] = float(np.mean(y[idx + 1] <= self.stop_eps))
        for c in range(2, max_c + 2):
            out[c] = max(out[c], out[c - 1])

        cs, ls = [], []
        for c in range(1, max_c + 1):
            cnt = int(np.sum(self._run[:n - 1] == c))
            p = out[c]
            if cnt >= min_n and 0.0 < p < 1.0:
                cs.append(np.log(c))
                ls.append(np.log(p / (1.0 - p)))
        tail = out[max_c]
        if len(cs) >= 2:
            b, a = np.polyfit(np.asarray(cs), np.asarray(ls), 1)
            b = max(float(b), 0.0)
            for c in range(max_c + 1, n + 1):
                z = float(a) + b * np.log(c)
                tail = max(tail, 1.0 / (1.0 + np.exp(-z)))
                out[c] = min(tail, 0.99)
        else:
            out[max_c + 1:] = tail
        return out

    def _survival_feature(self, run) -> np.ndarray:
        if self._surv is None:
            return np.zeros_like(np.asarray(run, dtype=np.float64))
        g = self._surv[np.minimum(np.asarray(run, dtype=np.int64), len(self._surv) - 1)]
        return np.clip(g, 0.0, 0.99)

    @staticmethod
    def _parse_action(raw, n: int) -> np.ndarray:
        """策略动作固定为 [delta, p_off]；对旧的一维策略给出明确错误，避免静默错跑。"""
        a = np.asarray(raw, dtype=np.float64)
        if a.ndim == 1:
            if n == 1 and a.size == 2:
                a = a.reshape(1, 2)
            else:
                raise ValueError(f"v3 策略必须输出 shape=(N,2) 的 [delta,p_off]，实得 {a.shape}")
        a = a.reshape(n, -1)
        if a.shape[1] != 2:
            raise ValueError(f"v3 策略必须输出 2 维动作 [delta,p_off]，实得 {a.shape}")
        return a

    def rollout(self, policy: Callable[[np.ndarray], np.ndarray], starts) -> Dict[str, np.ndarray]:
        """多个起报日并行执行 H 步预测。"""
        starts = np.asarray(starts, dtype=np.int64)
        N, H, L = len(starts), self.H, self.L
        off = self.delay
        win = self.y[starts[:, None] + np.arange(-L, 0)[None, :] - off].copy()
        S = np.zeros((N, H + 1, self.state_dim), np.float32)
        A = np.zeros((N, H, 2), np.float64)
        P = np.zeros((N, H), np.float64)
        P_ON = np.zeros((N, H), np.float64)
        BASE = np.zeros((N, H), np.float64)
        OFFP = np.zeros((N, H), np.float64)
        OFF = np.zeros((N, H), dtype=bool)
        SURV = np.zeros((N, H), np.float64)

        run = self._run[starts - 1 - off].astype(np.int64).copy()
        anchor0 = self.y[starts - 1 - off]
        for k in range(H + 1):
            S[:, k, :L] = win
            S[:, k, L] = k / H
            S[:, k, L + 1] = self._stop_feature(run)
            S[:, k, L + 2] = self._survival_feature(run)
            if k == H:
                break

            act = self._parse_action(policy(S[:, k]), N)
            delta = np.clip(act[:, 0], -np.inf, np.inf)
            p_off = np.clip(act[:, 1], 0.0, 1.0)
            A[:, k, 0] = delta
            A[:, k, 1] = p_off
            OFFP[:, k] = p_off
            SURV[:, k] = self._survival_feature(run)

            # 残差基准：首步相对起报时刻实测值，其后相对"起报锚点与上一步预测各半"。
            base = anchor0 if k == 0 else 0.5 * anchor0 + 0.5 * P[:, k - 1]
            BASE[:, k] = base
            # ON 分支保持在状态阈值之上，确保 off_head 状态与最终负荷状态一致。
            on_floor = self.z0 + max(float(self.cfg.on_eps), 1e-6)
            p_on = np.clip(base + delta, on_floor, self.p_max)
            P_ON[:, k] = p_on

            off = p_off >= self.off_threshold
            OFF[:, k] = off

            if self.gate_mode == 'hurdle':
                # 温度锐化的可微 hurdle 门控：p_off 超过阈值后预测快速趋近停机水平
                gate = 1.0 / (1.0 + np.exp(-(p_off - self.off_threshold) / self.gate_temp))
                pred = self.z0 + (1.0 - gate) * (p_on - self.z0)
            else:
                pred = p_on

            P[:, k] = np.clip(
                pred,
                self.z0,
                self.p_max
            )

            # 关键：使用模型自己的状态预测推进 run，而不是把起报时刻工况锁死整个 H 步。
            run = np.where(off, run + 1, 0).astype(np.int64)
            win = np.concatenate([win[:, 1:], P[:, k:k + 1]], 1)

        Y = self.truth[starts[:, None] + np.arange(H)[None, :]]
        return dict(starts=starts, S=S, A=A, P=P, P_ON=P_ON, BASE=BASE, Y=Y,
                    OFFP=OFFP, OFF=OFF, SURV=SURV, anchors=anchor0.copy(),
                    hist=self.truth.copy())  # 本段真值序列（MASE 尺度 / 爬坡限值参考）

    def transitions(self, ro: Dict[str, np.ndarray]):
        """把轨迹展开成 TD3 转移；动作维度为 2：[delta,p_off]。"""
        R, parts = lcar(ro['P'], ro['Y'], ro['anchors'], self.cfg, off_prob=ro.get('OFFP'))
        N, H = R.shape
        D = np.zeros((N, H))
        D[:, -1] = 1.0
        flat = lambda x: x.reshape(N * H, -1)
        return dict(s=flat(ro['S'][:, :H]), s2=flat(ro['S'][:, 1:]), a=flat(ro['A']), r=flat(R),
                    d=flat(D), y=flat(ro['Y']), yp=flat(parts['prev_true']),
                    w=flat(parts['W']), om=flat(parts['Om']), base=flat(ro['BASE'])), R
