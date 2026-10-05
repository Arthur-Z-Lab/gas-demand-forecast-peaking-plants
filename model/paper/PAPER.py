"""CBR-TD3：chaos-aware Bayesian reinforcement learning framework with TD3 optimization
（混沌感知贝叶斯强化学习 + TD3 优化）的多步负荷预测智能体。

不使用“基础控制器 + RL 残差”结构。Actor 本身就是统一预测策略：
Aihara CCNN 编码多尺度内生序列，Bayesian policy 直接输出 [delta, p_off]，
TD3 在递归预测 MDP 上优化该策略。每个 episode 固定一次 Bayesian 后验权重采样，
保证 H 步策略随机性在轨迹内一致。

训练完全基于历史运行数据：环境由历史序列构造，智能体不与燃气电厂或调度系统在线交互；
部署时只需当前历史窗口，即可连续输出未来 H 日负荷及其预测区间。
"""
import math
import pickle
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.paper.reward import make_reward_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_reward(horizon: int, overrides: Optional[dict] = None):
    return make_reward_config(horizon, overrides)


ACCEPTED_KWARGS = frozenset((
    'learning_rate_actor', 'learning_rate_critic', 'gamma', 'tau', 'policy_noise',
    'noise_clip', 'policy_freq', 'batch_size', 'memory_size', 'grad_clip',
    'n_chaos', 'chaos_kf', 'chaos_kr', 'chaos_alpha', 'chaos_eps', 'chaos_w_scale',
    'chaos_learn_dynamics', 'n_actor_hidden', 'n_critic_hidden', 'activation',
    'bayes_prior_sigma', 'bayes_rho_init', 'br_alpha',
    'rho_lr_scale', 'chaos_lr_scale',
    'ablate_bayes', 'ablate_chaos',
    'gate_mode',
    'explore_sigma',
    'anchor_weight', 'q_alpha', 'off_loss_weight', 'off_pos_weight', 'lam_action',
    'off_switch_weight', 'log_freq', 'device',
    'lam_delta', 'phys_low',
    'off_start_prior', 'off_stay_prior',
    'off_threshold', 'gate_temp',
    'point_scale',
))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _act(name: str) -> nn.Module:
    table = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU, 'elu': nn.ELU,
             'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}
    if name not in table:
        raise ValueError(f"激活函数 '{name}' 不支持；可选 {sorted(table)}")
    return table[name]()


def _inv_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError(f"softplus 反函数要求 x > 0，实得 {x}")
    return float(math.log(math.expm1(x)))


def _logit(p: float) -> float:
    return float(math.log(p / (1.0 - p)))


class ChaoticNeuronLayer(nn.Module):
    """Aihara 混沌神经网络层（Aihara, Takabe & Toyoda, 1990, Phys. Lett. A 144: 333-340）。

    η_i(t+1) = k_f·η_i(t) + Σ_j W_ij x_j(t) + (V u_t)_i          反馈（含外部输入）
    ζ_i(t+1) = k_r·ζ_i(t) − α·x_i(t) + a_i                        不应性（refractoriness）
    x_i(t+1) = sigmoid((η_i(t+1) + ζ_i(t+1)) / ε)
    k_f, k_r ∈ (0,1)、α > 0、ε > ε_min 经约束参数化后可学习；初始化位于混沌区（λ_max > 0）。
    """

    EPS_MIN = 0.01

    def __init__(self, n_in: int, n: int, *, kf: float, kr: float, alpha: float, eps: float,
                 w_scale: float, learn_dynamics: bool):
        super().__init__()
        if not (0.0 < kf < 1.0 and 0.0 < kr < 1.0):
            raise ValueError(f"chaos_kf / chaos_kr 必须在 (0, 1) 内，实得 {kf}, {kr}")
        if alpha <= 0.0:
            raise ValueError(f"chaos_alpha 必须 > 0，实得 {alpha}")
        if eps <= self.EPS_MIN:
            raise ValueError(f"chaos_eps 必须 > {self.EPS_MIN}，实得 {eps}")
        self.n = int(n)
        self.V = nn.Linear(n_in, self.n)
        self.W = nn.Parameter(torch.randn(self.n, self.n) * w_scale / math.sqrt(self.n))
        self.a = nn.Parameter(torch.empty(self.n).uniform_(0.3, 0.7))
        dyn = [('_kf', _logit(kf)), ('_kr', _logit(kr)), ('_alpha', _inv_softplus(alpha)),
               ('_eps', _inv_softplus(eps - self.EPS_MIN))]
        for name, v in dyn:
            p = nn.Parameter(torch.tensor(v))
            p.requires_grad_(bool(learn_dynamics))
            setattr(self, name, p)

    @property
    def kf(self):
        return torch.sigmoid(self._kf)

    @property
    def kr(self):
        return torch.sigmoid(self._kr)

    @property
    def alpha(self):
        return F.softplus(self._alpha)

    @property
    def eps(self):
        return F.softplus(self._eps) + self.EPS_MIN

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = seq.shape
        kf, kr, al, inv_eps = self.kf, self.kr, self.alpha, 1.0 / self.eps
        Wt = self.W.t()
        drive = self.V(seq)
        eta = seq.new_zeros(B, self.n)
        zeta = seq.new_zeros(B, self.n)
        x = torch.sigmoid((eta + zeta) * inv_eps)
        acc = 0.0
        for t in range(L):
            eta = kf * eta + x @ Wt + drive[:, t]
            zeta = kr * zeta - al * x + self.a
            x = torch.sigmoid((eta + zeta) * inv_eps)
            acc = acc + x
        return x, acc / L

    @torch.no_grad()
    def lyapunov_exponent(self, steps: int = 3000, burn_in: int = 500, seed: int = 0) -> float:
        """自治动力学（无外部输入）的最大 Lyapunov 指数，Benettin 两轨道重归一化法，float64。"""
        g = torch.Generator().manual_seed(seed)
        W = self.W.detach().double().cpu()
        a = self.a.detach().double().cpu()
        d = self.dynamics()
        kf, kr, al, ep = d['kf'], d['kr'], d['alpha'], d['eps']

        def step(e, z):
            x = torch.sigmoid((e + z) / ep)
            return kf * e + W @ x, kr * z - al * x + a

        e = torch.zeros(self.n, dtype=torch.float64)
        z = (torch.rand(self.n, generator=g, dtype=torch.float64) - 0.5) * 0.2
        d0 = 1e-8
        v = torch.randn(2 * self.n, generator=g, dtype=torch.float64)
        v *= d0 / v.norm()
        e2, z2 = e + v[:self.n], z + v[self.n:]
        acc = 0.0
        for t in range(steps):
            e, z = step(e, z)
            e2, z2 = step(e2, z2)
            dv = torch.cat([e2 - e, z2 - z])
            dn = max(float(dv.norm()), 1e-300)
            if t >= burn_in:
                acc += math.log(dn / d0)
            dv *= d0 / dn
            e2, z2 = e + dv[:self.n], z + dv[self.n:]
        return acc / (steps - burn_in)

    def dynamics(self) -> Dict[str, float]:
        with torch.no_grad():
            return dict(kf=self.kf.item(), kr=self.kr.item(), alpha=self.alpha.item(), eps=self.eps.item())


class ChaoticEncoder(nn.Module):
    """多尺度内生 CCNN 编码器。

    每个历史日输入 6 个完全由单变量负荷构造的内生量（多尺度历史负荷信息）：
      [y_t, Δ1 y_t, Δ3 y_t, Δ7 y_t, Δ14 y_t, y_t − MA7, I(y_t≈0)]
    覆盖日/周/双周三个尺度与"相对 7 日水平"的偏离量，不引入任何外生变量或未来信息。
    """

    N_CH = 7

    def __init__(self, seq_len: int, n_aux: int, **kw):
        super().__init__()
        self.L, self.n_aux = int(seq_len), int(n_aux)
        self.cnn = ChaoticNeuronLayer(
            self.N_CH, kw['n_chaos'], kf=kw['chaos_kf'], kr=kw['chaos_kr'], alpha=kw['chaos_alpha'],
            eps=kw['chaos_eps'], w_scale=kw['chaos_w_scale'],
            learn_dynamics=kw.get('chaos_learn_dynamics', True))
        self.out_dim = 2 * int(kw['n_chaos']) + self.L + self.n_aux

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        win = s[:, :self.L]
        x_last, x_mean = self.cnn(load_channels(win, self.L))
        return torch.cat([x_last, x_mean, win, s[:, self.L:]], 1)


def load_channels(win: torch.Tensor, L: int) -> torch.Tensor:
    """把单变量历史窗口展开成多尺度内生通道：水平 / Δ1 / Δ3 / Δ7 / Δ14 / 7日均值偏离 / 停机指示。"""

    def lag_diff(k: int) -> torch.Tensor:
        d = torch.zeros_like(win)
        if L > k:
            d[:, k:] = win[:, k:] - win[:, :-k]
        return d

    ma7 = torch.zeros_like(win)
    if L >= 7:
        csum = torch.cumsum(win, 1)
        prev = torch.zeros_like(csum)
        prev[:, 7:] = csum[:, :-7]                 # prev[t] = Σ_{i<t-6} win[i]
        ma7 = (csum - prev) / 7.0                  # t<6 处为部分和，下面置零
        ma7 = ma7 * (torch.arange(L, device=win.device) >= 6).to(win.dtype)
    z = (win <= 1e-6).to(win.dtype)
    return torch.stack([win, lag_diff(1), lag_diff(3), lag_diff(7), lag_diff(14), win - ma7, z], -1)


class MlpEncoder(nn.Module):
    """消融用编码器：与 ChaoticEncoder 同输入、同输出维度，但用前馈网络替代混沌神经元层。

    仅用于消融实验（--ablate_chaos 1），用来隔离"内生混沌编码"这一模块的贡献。
    """

    def __init__(self, seq_len: int, n_aux: int, **kw):
        super().__init__()
        self.L, self.n_aux = int(seq_len), int(n_aux)
        self.n_chaos = int(kw['n_chaos'])
        self.n_ch = ChaoticEncoder.N_CH
        self.fc = nn.Sequential(
            nn.Linear(self.n_ch * self.L + self.n_aux, 2 * self.n_chaos), _act(kw['activation']),
            nn.Linear(2 * self.n_chaos, 2 * self.n_chaos), _act(kw['activation']))
        self.out_dim = 2 * self.n_chaos + self.L + self.n_aux

    def forward(self, s):
        win = s[:, :self.L]
        ch = load_channels(win, self.L).flatten(1)
        h = self.fc(torch.cat([ch, s[:, self.L:]], 1))
        return torch.cat([h, win, s[:, self.L:]], 1)


def make_encoder(seq_len: int, n_aux: int, **kw) -> nn.Module:
    """按配置返回内生编码器：默认混沌（CCNN），--ablate_chaos 1 时返回同维度 MLP。"""
    return MlpEncoder(seq_len, n_aux, **kw) if int(kw.get('ablate_chaos', 0)) else \
        ChaoticEncoder(seq_len, n_aux, **kw)


class BayesLinear(nn.Module):
    """w = μ + softplus(ρ)⊙ε（Blundell et al., 2015, ICML, Bayes by Backprop）。
    mode: 'sample' 每次前向重采样（训练用，局部重参数化）；'mean' 取后验均值；'fixed' 使用
    resample() 冻结的一组权重（推理/探索用，逐 episode 固定）。

    训练用**局部重参数化**而非直接采样权重：直接采样时 ∂w/∂ρ = ε·sigmoid(ρ)，
    ρ_init=-5 时该因子只有 0.0067，ρ 的梯度比 μ 小两个数量级且被零均值 ε 变成噪声主导，
    实测 2 万次更新后验 σ 只漂 3e-7（即"贝叶斯层冻结"）。局部重参数化把噪声搬到输出端：
        y = x·μ + b_μ + ε·sqrt(x²·σ_w² + σ_b²)
    同一线性层在数学上等价，但 ρ 的梯度量级恢复到与 μ 同级。
    """

    def __init__(self, n_in: int, n_out: int, prior_sigma: float, rho_init: float):
        super().__init__()
        self.prior_sigma = float(prior_sigma)
        self.w_mu = nn.Parameter(torch.empty(n_out, n_in))
        self.b_mu = nn.Parameter(torch.empty(n_out))
        self.w_rho = nn.Parameter(torch.full((n_out, n_in), float(rho_init)))
        self.b_rho = nn.Parameter(torch.full((n_out,), float(rho_init)))
        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        nn.init.uniform_(self.b_mu, -1 / math.sqrt(n_in), 1 / math.sqrt(n_in))
        self.n_params = n_out * n_in + n_out
        self.mode = 'sample'
        self.register_buffer('w_eps', torch.zeros(n_out, n_in), persistent=False)
        self.register_buffer('b_eps', torch.zeros(n_out), persistent=False)

    def resample(self):
        self.w_eps.normal_()
        self.b_eps.normal_()

    def forward(self, x):
        if self.mode == 'mean':
            return F.linear(x, self.w_mu, self.b_mu)
        ws, bs = F.softplus(self.w_rho), F.softplus(self.b_rho)
        if self.mode == 'sample':
            var = x.pow(2) @ ws.pow(2).t() + bs.pow(2)
            return F.linear(x, self.w_mu, self.b_mu) + torch.randn_like(var) * var.clamp_min(1e-12).sqrt()
        return F.linear(x, self.w_mu + self.w_eps * ws, self.b_mu + self.b_eps * bs)

    def kl(self) -> torch.Tensor:
        out = 0.0
        for mu, rho in ((self.w_mu, self.w_rho), (self.b_mu, self.b_rho)):
            s = F.softplus(rho).clamp_min(1e-8)
            out = out + (torch.log(self.prior_sigma / s)
                         + (s ** 2 + mu ** 2) / (2 * self.prior_sigma ** 2) - 0.5).sum()
        return out


class DeterministicLinear(nn.Module):
    """消融用策略层：与 BayesLinear 接口一致，但权重是点估计（无后验、无 KL）。

    仅用于消融实验（--ablate_bayes 1），用来隔离"贝叶斯策略参数化"的贡献。
    """

    def __init__(self, n_in: int, n_out: int, prior_sigma: float = 0.1, rho_init: float = -3.0):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.n_params = n_out * n_in + n_out

    # 与 BayesLinear 同名接口：Actor 初始化先验偏置时会用到 w_mu / b_mu
    @property
    def w_mu(self):
        return self.lin.weight

    @property
    def b_mu(self):
        return self.lin.bias

    def resample(self):
        pass

    def mode_set(self, mode):
        pass

    def forward(self, x):
        return self.lin(x)

    def kl(self):
        return torch.zeros((), device=self.lin.weight.device)


def make_policy_layer(ablate_bayes: int, n_in: int, n_out: int, prior_sigma: float, rho_init: float):
    """--ablate_bayes 1 时用确定性线性层替代贝叶斯层（消融用）。"""
    if int(ablate_bayes):
        return DeterministicLinear(n_in, n_out, prior_sigma, rho_init)
    return BayesLinear(n_in, n_out, prior_sigma, rho_init)


class Actor(nn.Module):
    """
    Actor of CBR-TD3（混沌编码器 + 贝叶斯策略头）。

    action[:, 0] = delta
    action[:, 1] = p_off

    状态头拆分为：
        p_shutdown = P(next=OFF | current=ON, s)
        p_stayoff  = P(next=OFF | current=OFF, s)

    根据当前递归状态选择最终 p_off。
    """

    def __init__(self, seq_len, n_aux, a_low, a_high, **kw):
        super().__init__()

        self.L = int(seq_len)
        self.a_low = float(a_low)
        self.a_high = float(a_high)

        self.encoder = make_encoder(seq_len, n_aux, **kw)

        h = int(kw['n_actor_hidden'])
        ps = float(kw['bayes_prior_sigma'])
        ri = float(kw['bayes_rho_init'])
        ab = int(kw.get('ablate_bayes', 0))

        # Bayesian shared policy trunk
        self.b1 = make_policy_layer(ab, self.encoder.out_dim, h, ps, ri)
        self.b2 = make_policy_layer(ab, h, h, ps, ri)

        # Bayesian load head
        self.load_head = make_policy_layer(ab, h, 1, ps, ri)

        # Bayesian semi-Markov regime heads
        self.shutdown_head = make_policy_layer(ab, h, 1, ps, ri)
        self.stayoff_head = make_policy_layer(ab, h, 1, ps, ri)

        start_prior = float(
            np.clip(kw.get('off_start_prior', 0.03), 1e-4, 1 - 1e-4)
        )
        stay_prior = float(
            np.clip(kw.get('off_stay_prior', 0.70), 1e-4, 1 - 1e-4)
        )

        # 初始时两个状态头只输出经验转移先验
        nn.init.zeros_(self.shutdown_head.w_mu)
        nn.init.constant_(
            self.shutdown_head.b_mu,
            _logit(start_prior)
        )

        nn.init.zeros_(self.stayoff_head.w_mu)
        # stayoff 头输出的是对训练段经验生存链 P(仍停|已停 c 天) 的 logit **修正量**，
        # 故基线置 0（见 components()），长停机外推时 p_off 会自然趋近 1。
        nn.init.constant_(self.stayoff_head.b_mu, 0.0)

        self.f1 = _act(kw['activation'])
        self.f2 = _act(kw['activation'])

        self.bayes = [
            self.b1,
            self.b2,
            self.load_head,
            self.shutdown_head,
            self.stayoff_head,
        ]

        self.n_bayes = sum(m.n_params for m in self.bayes)

    def set_mode(self, mode: str):
        for m in self.bayes:
            m.mode = mode

    def resample(self):
        for m in self.bayes:
            m.resample()

    def trunk(self, s):
        z = self.encoder(s)
        z = self.f1(self.b1(z))
        z = self.f2(self.b2(z))
        return z

    def components(self, s):
        z = self.trunk(s)

        # ---------- continuous load ----------
        t = torch.tanh(self.load_head(z))

        delta = (
            self.a_low
            + (self.a_high - self.a_low)
            * (t + 1.0)
            * 0.5
        )

        # ---------- regime probabilities ----------
        shutdown_logit = self.shutdown_head(z)
        stayoff_logit = self.stayoff_head(z)

        # 状态里第 L+2 维是 train 段经验生存概率 p(c)=P(次日仍停|已连续停机 c 天)。
        # 以它为 logit 基线：长停机时基线本身趋近 1，头只需学习修正量，
        # 从而把训练段未出现过的超长停机（如 56 天）外推为 p_off≈1。
        surv = s[:, self.L + 2:self.L + 3].clamp(1e-4, 1.0 - 1e-4)
        stayoff_logit = stayoff_logit + torch.log(surv) - torch.log1p(-surv)

        p_shutdown = torch.sigmoid(shutdown_logit)
        p_stayoff = torch.sigmoid(stayoff_logit)

        # state:
        # [window L, horizon, run_feature, survival_feature]
        #
        # run_feature > 0 means current recursive regime is OFF.
        current_off = (
            s[:, self.L + 1:self.L + 2] > 1e-8
        ).to(z.dtype)

        p_off = (
            (1.0 - current_off) * p_shutdown
            + current_off * p_stayoff
        )

        return (
            delta,
            shutdown_logit,
            stayoff_logit,
            p_shutdown,
            p_stayoff,
            p_off,
        )

    def forward(self, s):
        delta, _, _, _, _, p_off = self.components(s)

        # Critic / environment仍只看到二维动作
        return torch.cat([delta, p_off], dim=1)

    def kl(self):
        return sum(m.kl() for m in self.bayes) / self.n_bayes

    def posterior_sigma(self) -> float:
        with torch.no_grad():
            sigmas = [F.softplus(m.w_rho).flatten() for m in self.bayes if hasattr(m, 'w_rho')]
            if not sigmas:      # 消融：确定性策略层没有变分参数
                return 0.0
            return float(torch.cat(sigmas).mean())


class Critic(nn.Module):
    """价值网络 = 混沌神经网络编码器 + 二维动作 [delta,p_off]；双 Q。"""

    def __init__(self, seq_len, n_aux, action_dim=2, **kw):
        super().__init__()
        h = kw['n_critic_hidden']
        self.enc1 = make_encoder(seq_len, n_aux, **kw)
        self.enc2 = make_encoder(seq_len, n_aux, **kw)
        d = self.enc1.out_dim + int(action_dim)

        def head():
            return nn.Sequential(nn.Linear(d, h), _act(kw['activation']),
                                 nn.Linear(h, h), _act(kw['activation']), nn.Linear(h, 1))
        self.h1, self.h2 = head(), head()

    def forward(self, s, a):
        return (self.h1(torch.cat([self.enc1(s), a], 1)),
                self.h2(torch.cat([self.enc2(s), a], 1)))

    def q1(self, s, a):
        return self.h1(torch.cat([self.enc1(s), a], 1))


class ReplayBuffer:
    KEYS = ('s', 'a', 'r', 's2', 'd', 'y', 'yp', 'w', 'om', 'base')

    def __init__(self, capacity: int, state_dim: int, action_dim: int = 2):
        self.cap = int(capacity)
        self.s = np.zeros((self.cap, state_dim), np.float32)
        self.s2 = np.zeros((self.cap, state_dim), np.float32)
        self.a = np.zeros((self.cap, int(action_dim)), np.float32)
        for k in ('r', 'd', 'y', 'yp', 'w', 'om', 'base'):
            setattr(self, k, np.zeros((self.cap, 1), np.float32))
        self.ptr = self.size = 0

    def add_batch(self, **t):
        n = len(t['a'])
        idx = (self.ptr + np.arange(n)) % self.cap
        for k in self.KEYS:
            getattr(self, k)[idx] = np.asarray(t[k], np.float32).reshape(n, -1)
        self.ptr = int((self.ptr + n) % self.cap)
        self.size = min(self.size + n, self.cap)

    def sample(self, n: int, dev):
        idx = np.random.randint(0, self.size, size=n)
        return {k: torch.from_numpy(getattr(self, k)[idx]).to(dev) for k in self.KEYS}

    def __len__(self):
        return self.size


class PAPER:

    def __init__(self, n_features, seq_len, n_aux, horizon, a_low, a_high, **kw):
        self.device = kw.get('device', device)
        self.L, self.n_aux, self.H = int(seq_len), int(n_aux), int(horizon)
        assert self.L + self.n_aux == int(n_features)
        self.a_low, self.a_high = float(a_low), float(a_high)
        self.phys_low = float(kw['phys_low'])
        self.action_range = self.a_high - self.a_low
        self.gamma, self.tau = float(kw['gamma']), float(kw['tau'])
        self.batch_size, self.policy_freq = int(kw['batch_size']), int(kw['policy_freq'])
        half = self.action_range / 2.0
        self.policy_noise, self.noise_clip = float(kw['policy_noise']) * half, float(kw['noise_clip']) * half
        self.grad_clip = kw['grad_clip']
        self.gate_mode = str(kw.get('gate_mode', 'hurdle'))
        self.br_alpha = float(kw['br_alpha'])
        self.anchor_weight, self.q_alpha = float(kw['anchor_weight']), float(kw['q_alpha'])
        self.off_loss_weight = float(kw['off_loss_weight'])
        self.off_pos_weight = float(kw['off_pos_weight'])
        self.off_switch_weight = float(kw['off_switch_weight'])
        # 动作收缩：把 delta 拉向 0（=停留在 base 水平）。负荷在多数日子几乎不变时，
        # 无此项会让网络在“平段”上无端产生位移，实测平段 MAE 是持续法的 6 倍。
        self.lam_action = float(kw.get('lam_action', 0.0))
        self.lam_delta = float(kw['lam_delta'])
        self.point_scale = float(kw.get('point_scale', 6.0))
        self.log_freq = int(kw['log_freq'])
        self.action_dim = 2

        self.actor = Actor(self.L, self.n_aux, a_low, a_high, **kw).to(self.device)
        self.actor_target = Actor(self.L, self.n_aux, a_low, a_high, **kw).to(self.device)
        self.critic = Critic(self.L, self.n_aux, action_dim=self.action_dim, **kw).to(self.device)
        self.critic_target = Critic(self.L, self.n_aux, action_dim=self.action_dim, **kw).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in list(self.actor_target.parameters()) + list(self.critic_target.parameters()):
            p.requires_grad_(False)
        self.actor_target.set_mode('mean')
        # 变分参数 ρ 与混沌动力学参数走单独参数组：它们经过 sigmoid/softplus 的雅可比很小
        # （∂kf/∂_kf = kf(1-kf) ≈ 0.16，∂w/∂ρ ≈ 0.007），与主干共用 1e-4 学习率时净漂移
        # 只有 1e-3 量级，实测等于"名义可学、实际冻结"。这里给它们放大学习率。
        lr_a = float(kw['learning_rate_actor'])
        rho_scale = float(kw.get('rho_lr_scale', 20.0))
        chaos_scale = float(kw.get('chaos_lr_scale', 10.0))
        rest_p, rho_p, chaos_p = [], [], []
        for name, p in self.actor.named_parameters():
            if name.endswith('w_rho') or name.endswith('b_rho'):
                rho_p.append(p)
            elif name.split('.')[-1] in ('_kf', '_kr', '_alpha', '_eps'):
                chaos_p.append(p)
            else:
                rest_p.append(p)
        self.actor_opt = optim.Adam([
            {'params': rest_p, 'lr': lr_a},
            {'params': rho_p, 'lr': lr_a * rho_scale},
            {'params': chaos_p, 'lr': lr_a * chaos_scale},
        ])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=kw['learning_rate_critic'])

        # 探索噪声：原版 TD3 的高斯动作噪声（标准差以动作半宽为单位）。
        # 消融结论：对比过 OU/均匀/截断/重尾 t/衰减共 6 种形式，高斯综合最优（见 README）。
        self.explore_noise = float(kw.get('explore_sigma', 0.1)) * half
        self.memory = ReplayBuffer(kw['memory_size'], int(n_features), self.action_dim)
        self.total_it = 0
        self.logs = dict(train=[], chaos=[])
        self.off_threshold = float(kw.get('off_threshold', 0.50))
        self.gate_temp = float(kw.get('gate_temp', 0.08))

        if self.gate_temp <= 0:
            raise ValueError("gate_temp 必须 > 0")

    # ---------- 动作 ----------
    def begin_episodes(self, n: int, progress: float):
        """一轮并行 episode 开始：冻结采样一组后验策略权重（组内 H 步策略随机性一致）。"""
        self.actor.resample()

    def explore(self, states: np.ndarray) -> np.ndarray:
        # 标准 TD3 探索：对连续动作 Δ 加高斯噪声；p_off 由显式状态监督学习，不随机扰动。
        a = self.act_batch(states, mode='fixed')
        a[:, 0] = np.clip(a[:, 0] + np.random.normal(0.0, self.explore_noise, size=len(a)),
                          self.a_low, self.a_high)
        a[:, 1] = np.clip(a[:, 1], 0.0, 1.0)
        return a

    @torch.no_grad()
    def act_batch(self, states: np.ndarray, mode: str = 'mean') -> np.ndarray:
        s = torch.as_tensor(np.asarray(states, np.float32), device=self.device)
        self.actor.set_mode(mode)
        out = self.actor(s).cpu().numpy().astype(np.float64)
        self.actor.set_mode('sample')
        return out

    def resample_policy(self):
        self.actor.resample()

    @torch.no_grad()
    def q_values(self, states, actions, chunk: int = 4096):
        q1s, qms = [], []
        for i in range(0, len(states), chunk):
            s = torch.as_tensor(np.asarray(states[i:i + chunk], np.float32), device=self.device)
            a = torch.as_tensor(np.asarray(actions[i:i + chunk], np.float32), device=self.device)
            q1, q2 = self.critic(s, a)
            q1s.append(q1.squeeze(-1).cpu().numpy())
            qms.append(torch.min(q1, q2).squeeze(-1).cpu().numpy())
        return np.concatenate(q1s).astype(np.float64), np.concatenate(qms).astype(np.float64)

    # ---------- 学习 ----------
    def store(self, **t):
        self.memory.add_batch(**t)

    def _load_loss(self, delta, p_off, b):
        """
        最终负荷使用 temperature-sharpened hurdle gate。

        目的：
        低 p_off 时几乎不影响正常负荷；
        高 p_off 时快速趋近停机水平。
        """

        y, yp = b['y'], b['yp']

        pred_on = torch.clamp(
            b['base'] + delta,
            min=self.phys_low
        )

        if self.gate_mode == 'hurdle':
            gate = torch.sigmoid(
                (p_off - self.off_threshold)
                / self.gate_temp
            )

            pred = (
                    self.phys_low
                    + (1.0 - gate)
                    * (pred_on - self.phys_low)
            )
        else:
            # 无门控：点预测直接取正负荷分支（与 env.gate_mode='none' 一致）
            pred = pred_on

        prev_hat = b['s'][:, self.L - 1:self.L]

        point = self.point_scale * b['w'] * (pred - y) ** 2

        delta_loss = self.lam_delta * (
                (pred - prev_hat)
                - (y - yp)
        ).pow(2)

        return (
                b['om'] * (point + delta_loss)
        ).mean()

    def _state_loss(
            self,
            shutdown_logit,
            stayoff_logit,
            b
    ):
        """
        两个条件状态任务分别训练：

        shutdown head:
            P(OFF_t | ON_{t-1})

        stay-off head:
            P(OFF_t | OFF_{t-1})

        ON→OFF 和 OFF→ON 不再混在一个 BCE 中。
        """

        target_off = (
                b['y']
                <= self.phys_low + 1e-6
        ).float()

        prev_off = (
                b['yp']
                <= self.phys_low + 1e-6
        ).float()

        on_mask = 1.0 - prev_off
        off_mask = prev_off

        # =====================================================
        # 1. ON -> ?
        # =====================================================
        start_bce = F.binary_cross_entropy_with_logits(
            shutdown_logit,
            target_off,
            reduction='none'
        )

        # ON->OFF 很稀少，重点提高 shutdown positive 样本
        start_weight = (
                1.0
                + (self.off_pos_weight - 1.0)
                * target_off
        )

        start_loss = (
                             start_bce
                             * start_weight
                             * on_mask
                     ).sum() / on_mask.sum().clamp_min(1.0)

        # =====================================================
        # 2. OFF -> ?
        # =====================================================
        stay_bce = F.binary_cross_entropy_with_logits(
            stayoff_logit,
            target_off,
            reduction='none'
        )

        # 对 OFF->ON restart 样本额外加权
        restart = 1.0 - target_off

        stay_weight = (
                1.0
                + (self.off_switch_weight - 1.0)
                * restart
        )

        stay_loss = (
                            stay_bce
                            * stay_weight
                            * off_mask
                    ).sum() / off_mask.sum().clamp_min(1.0)

        # 防止训练集 OFF 少导致其中一个任务压倒另一个
        return 0.5 * start_loss + 0.5 * stay_loss

    def learn(self) -> Optional[torch.Tensor]:
        """一次 TD3 更新；返回本次策略目标 −E[Q]（张量，未同步到 CPU），非策略更新步返回 None。"""
        if len(self.memory) < self.batch_size:
            return None
        self.total_it += 1
        b = self.memory.sample(self.batch_size, self.device)
        with torch.no_grad():
            a2 = self.actor_target(b['s2'])
            # TD3 target policy smoothing 只作用于 delta；状态概率保持在 [0,1]。
            eps = (torch.randn_like(a2[:, :1]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = torch.cat([(a2[:, :1] + eps).clamp(self.a_low, self.a_high),
                            a2[:, 1:2].clamp(0.0, 1.0)], dim=1)
            tq1, tq2 = self.critic_target(b['s2'], a2)
            target = b['r'] + (1.0 - b['d']) * self.gamma * torch.min(tq1, tq2)
        q1, q2 = self.critic(b['s'], b['a'])
        c_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        c_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()
        if self.total_it % self.policy_freq:
            return None

        self.actor.set_mode('sample')
        (
            delta,
            shutdown_logit,
            stayoff_logit,
            p_shutdown,
            p_stayoff,
            p_off,
        ) = self.actor.components(b['s'])



        a = torch.cat([delta, p_off], dim=1)
        q = self.critic.q1(b['s'], a)
        pg = -q.mean()
        kl = self.actor.kl()
        load_loss = self._load_loss(delta, p_off, b) if self.anchor_weight > 0 \
            else torch.zeros((), device=self.device)
        state_loss = (
            self._state_loss(
                shutdown_logit,
                stayoff_logit,
                b
            )
            if self.off_loss_weight > 0
            else torch.zeros((), device=self.device)
        )
        lam = self.q_alpha / torch.clamp(q.abs().mean().detach(), min=0.05)
        shrink = self.lam_action * delta.pow(2).mean() if self.lam_action > 0 else 0.0
        a_loss = (lam * pg + self.anchor_weight * load_loss + self.off_loss_weight * state_loss
                  + self.br_alpha * kl + shrink)
        self.actor_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()
        self._soft(self.actor_target, self.actor)
        self._soft(self.critic_target, self.critic)

        if (self.total_it // self.policy_freq) % self.log_freq == 0:
            self.logs['train'].append(dict(
                step=self.total_it, q_buffer=q1.mean().item(), td_target=target.mean().item(),
                q_actor=q.mean().item(), anchor=load_loss.item(), state_bce=state_loss.item(), kl=kl.item(),
                shrink=float(shrink.detach()) if torch.is_tensor(shrink) else float(shrink),
                post_sigma=self.actor.posterior_sigma(), critic_loss=c_loss.item()))
            cnn = getattr(self.actor.encoder, 'cnn', None)
            if cnn is not None:      # 消融（MLP 编码器）时没有混沌动力学可记录
                self.logs['chaos'].append(dict(step=self.total_it, **cnn.dynamics()))
        return pg.detach()

    @torch.no_grad()
    def _soft(self, tgt, src):
        for t, c in zip(tgt.parameters(), src.parameters()):
            t.data.mul_(1 - self.tau).add_(c.data, alpha=self.tau)

    # ---------- 状态与保存 ----------
    def actor_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}

    def load_actor_state(self, sd):
        self.actor.load_state_dict(sd)

    def lyapunov(self) -> Dict[str, float]:
        cnn = getattr(self.actor.encoder, 'cnn', None)
        if cnn is None:      # 消融：编码器换成 MLP 时没有混沌动力学
            return dict(lyap_actor=float('nan'), lyap_critic=float('nan'))
        return dict(lyap_actor=cnn.lyapunov_exponent(),
                    lyap_critic=self.critic.enc1.cnn.lyapunov_exponent(),
                    **{f"chaos_{k}": v for k, v in cnn.dynamics().items()})

    def save_model(self, path: Path):
        torch.save(self.actor.state_dict(), path)

    def save_logs(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self.logs, f)
