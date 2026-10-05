"""命令行配置。

设计约定：**本文件只暴露"可调数值参数"**，不暴露任何开关/模式选择。
所有结构性与策略性选择（残差建模、停机生存先验、hurdle 门控、线性先验混合、
早停、损失族、残差基准……）都已固化在 ``BEST_CONFIG`` 中并设为当前最优配置，
由 ``apply_best_config`` 注入到 argparse 结果上，因此 run.py 其余代码无需改动。

好处：调参只需改数值；不会因为漏传某个开关而跑到非最优结构上，
配置指纹（trials_*.csv 的 config 字段）也不会再随开关组合漂移。
"""
import argparse
from pathlib import Path
from typing import Dict, List

from model.paper.PAPER import ACCEPTED_KWARGS

# 本文件位于项目根目录（与 run.py 同级）
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ARGS: List[str] = []
REWARD_ARGS: List[str] = []

# 残差模式动作界（训练段经验值；非残差模式不使用）
DEFAULT_DELTA_LOW, DEFAULT_DELTA_HIGH = -0.80, 0.80

# ---------------------------------------------------------------------------
# 固化的最优配置：不通过命令行暴露，避免"开关组合"造成非最优实验
# ---------------------------------------------------------------------------
# 仍需注入到 args 上的运行时开关（其余结构性选择已直接固化在各自模块里）：
#   残差基准/门控           -> src/env.py（rollout 内固定实现）
#   停机生存先验锚定、混沌动力学可学习 -> model/paper/PAPER.py 的默认值
#   点损失族(mse)、线性先验混合(验证段选权重) -> model/paper/reward.py、run.py
BEST_CONFIG: Dict[str, object] = dict(
    stop_survival=True,   # 停机生存先验作为状态输入
    early_stop=True,      # 验证段早停 + 回滚到验证最优
    no_figures=False,     # 始终输出图
)

__all__ = ["PROJECT_ROOT", "BEST_CONFIG", "apply_best_config", "build_parser",
           "model_kwargs", "reward_overrides", "DEFAULT_DELTA_LOW", "DEFAULT_DELTA_HIGH"]


def apply_best_config(args) -> None:
    """把固化的最优配置注入 argparse 结果（覆盖那些已不再暴露的开关属性）。"""
    for k, v in BEST_CONFIG.items():
        setattr(args, k, v)


def _m(g, *flags, **kw):
    a = g.add_argument(*flags, **kw)
    if a.dest not in MODEL_ARGS:
        MODEL_ARGS.append(a.dest)
    return a


def _r(g, *flags, **kw):
    a = g.add_argument(*flags, **kw)
    if a.dest not in REWARD_ARGS:
        REWARD_ARGS.append(a.dest)
    return a


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CBR-TD3：多步燃气调峰负荷预测"
                    "（结构固化为最优配置，命令行只保留数值参数）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = ap.add_argument_group('数据与协议')
    g.add_argument('--data', default='data/HDQS.xlsx')
    g.add_argument('--col', default='HDQS')
    g.add_argument('--train_ratio', type=float, default=0.80)
    g.add_argument('--val_ratio', type=float, default=0.10)
    g.add_argument('--n_folds', type=int, default=1,
                   help='walk-forward 折数；1=单折（test 取末尾 1-train_ratio-val_ratio），'
                        'K>1 时训练窗口 expanding、每折测试段长度 = (1-train_ratio-val_ratio)×N')
    g.add_argument('--only_fold', type=int, default=0,
                   help='只跑第 k 折（1 基；0=全部折）。折划分、种子、超参仍由 --n_folds 决定，'
                        '因此协议指纹不变、已完成的折会被自动跳过；'
                        '用途：把"折"作为并行作业粒度，避免每个 (步长,种子) 作业串行跑 5 折的长尾')
    g.add_argument('--seq_len', type=int, default=24, help='历史窗口 L（日）')
    g.add_argument('--horizon', type=int, default=7, help='预测步长 H（日）')
    g.add_argument('--seeds', default='42', help='随机种子，逗号分隔')
    g.add_argument('--outdir', default=str(PROJECT_ROOT / 'output' / 'main' / 'HDQS'))

    g = ap.add_argument_group('训练预算与早停')
    g.add_argument('--total_episodes', type=int, default=40000, help='训练预算：累计执行的 episode 数')
    g.add_argument('--n_parallel', type=int, default=64, help='每轮并行执行的 episode 数')
    g.add_argument('--utd', type=float, default=0.25, help='每条新转移对应的梯度更新次数')
    g.add_argument('--warmup_episodes', type=int, default=0, help='开头随机动作采集的 episode 数')
    g.add_argument('--val_every', type=int, default=500, help='每多少 episode 在验证段评估一次')
    g.add_argument('--patience', type=int, default=8, help='连续多少次验证无改进则早停')
    g.add_argument('--min_episodes', type=int, default=4000, help='早停前至少训练的 episode 数')

    g = ap.add_argument_group('门控与预测范围')
    g.add_argument('--off_threshold', type=float, default=0.50, help='p_off≥阈值判定停机（软门控中心）')
    g.add_argument('--gate_temp', type=float, default=0.08, help='门控温度；越小越接近硬门控')
    g.add_argument('--p_max', type=float, default=1.50, help='预测上界（标幺），防止多步累积发散')
    g.add_argument('--gate_mode', default='hurdle', choices=['hurdle', 'none'],
                   help='点预测门控：hurdle=p_off 软门控（默认）；none=直接用正负荷分支')
    g.add_argument('--val_metric', default='mae', choices=['mae', 'balanced'],
                   help='早停/选检查点的验证准则：mae=整体 MAE（与主指标一致）；balanced=ON/OFF/切换等权')
    # 消融结论（HDQS，折4-5，seed42，2 次训练/单元）：开启线性先验门控混合使 MAE
    # H=1 −1.8%、H=7 −6.0%、H=15 −3.4%（平均 −3.7%）→ 默认开启
    g.add_argument('--use_prior', type=int, default=1,
                   help='1=启用线性先验门控混合（默认开；消融显示对精度有正贡献）')
    g.add_argument('--blend_metric', default='mae', choices=['mae', 'mse'],
                   help='线性先验混合权重 w 的验证段选择准则（仅 use_prior=1）')
    g.add_argument('--ridge_alpha', type=float, default=1.0, help='线性先验 Ridge 正则强度（仅 use_prior=1）')
    g.add_argument('--ridge_lags', type=int, default=14, help='线性先验滞后阶数（仅 use_prior=1）')
    # 论文同时报告两套区间：PICP/PINAW 取 --pi_method 指定的主口径，Bayes_*/Cal_* 全量输出。
    # 正文与附录采用验证段共形校准作为主口径（见论文表 B4），贝叶斯后验采样区间作为对照。
    g.add_argument('--pi_method', default='conformal', choices=['bayes', 'conformal'],
                   help='主预测区间口径：conformal=验证段共形校准（默认，覆盖≈标称）；'
                        'bayes=贝叶斯后验采样（两套指标都会输出）')

    g = ap.add_argument_group('预测区间')
    # 与论文表 B4 一致：共形校准水平与报告区间水平均为 0.95。
    g.add_argument('--pi_level', type=float, default=0.95,
                   help='主预测区间的标称水平（论文取 0.95；1-α_PI）')
    g.add_argument('--mc_samples', type=int, default=100)

    g = ap.add_argument_group('奖励（LCAR，标幺口径）')
    _r(g, '--reward_type', default='lcar',
       choices=['lcar', 'quantile', 'pinball', 'wmae', 'pwmse'],
       help='奖励函数：lcar=本文方法；quantile=分位数损失(τ=0.5)；pinball=Pinball损失(τ 见下)；'
            'wmae=加权MAE；pwmse=峰值加权MSE（后四者为消融对照，整体替换 LCAR）')
    _r(g, '--pinball_tau', type=float, default=0.9, help='Pinball 损失的分位数 τ')
    _r(g, '--point_scale', type=float, default=6.0, help='点预测平方误差主项倍率')
    _r(g, '--beta_change', type=float, default=1.0)
    _r(g, '--change_ref', type=float, default=0.10)
    _r(g, '--beta_peak', type=float, default=0.5)
    _r(g, '--y_peak', type=float, default=0.70)
    _r(g, '--kappa', type=float, default=12.0)
    _r(g, '--lam_delta', type=float, default=0.08)
    _r(g, '--lam_dir', type=float, default=0.01)
    _r(g, '--dir_eps', type=float, default=0.02)
    _r(g, '--lam_state', type=float, default=0.15, help='运行状态错判惩罚（对齐停机工况漂移）')
    _r(g, '--lam_growth', type=float, default=0.15)
    _r(g, '--growth_margin', type=float, default=0.02)
    # 可选奖励项（--reward_extras 1 时生效；用于消融其对精度的贡献）
    # 消融结论：quality+sparse+η 三项对 MAE 平均影响 ≈0（H=1/7 −0.6%，H=15 +2.2%）→ 默认关
    _r(g, '--reward_extras', type=int, default=0,
       help='1=启用可选奖励项（分段质量奖励 + episode 末步稀疏奖励 + 步数加权 η）')
    _r(g, '--eta_h', type=float, default=0.0, help='步数加权系数（需 reward_extras=1）')
    _r(g, '--quality_weight', type=float, default=0.05)
    _r(g, '--quality_ref', type=float, default=0.08)
    _r(g, '--quality_temp', type=float, default=0.04)
    _r(g, '--sparse_weight', type=float, default=0.10)
    _r(g, '--sparse_ref', type=float, default=0.10)
    _r(g, '--sparse_temp', type=float, default=0.04)
    _r(g, '--clip_reward', type=float, default=5.0)
    g.add_argument('--on_eps', type=float, default=0.01, help='停机判定占位阈值（诊断用）')

    g = ap.add_argument_group('TD3')
    _m(g, '--lr_actor', dest='learning_rate_actor', type=float, default=1e-4)
    _m(g, '--lr_critic', dest='learning_rate_critic', type=float, default=3e-4)
    _m(g, '--gamma', type=float, default=0.99, help='折扣因子')
    _m(g, '--tau', type=float, default=0.005)
    _m(g, '--policy_noise', type=float, default=0.1)
    _m(g, '--noise_clip', type=float, default=0.2)
    _m(g, '--policy_freq', type=int, default=2)
    _m(g, '--batch_size', type=int, default=512)
    _m(g, '--memory_size', type=int, default=100000)
    _m(g, '--grad_clip', type=float, default=1.0)
    _m(g, '--anchor_weight', type=float, default=2.0, help='最终负荷监督项权重')
    _m(g, '--q_alpha', type=float, default=0.10, help='TD3 策略项尺度 λ=q_alpha/max(mean|Q|,0.05)')
    _m(g, '--lam_action', type=float, default=0.1, help='动作收缩权重（把 Δ 拉向 0，抑制平段位移）')
    _m(g, '--off_loss_weight', type=float, default=1.0, help='运行状态分类 BCE 权重')
    _m(g, '--off_pos_weight', type=float, default=8.0, help='BCE 停机类正样本权重')
    _m(g, '--off_switch_weight', type=float, default=3.0, help='启停切换样本的 BCE 额外权重')

    g = ap.add_argument_group('混沌神经网络（Aihara 模型）')
    _m(g, '--n_chaos', type=int, default=128, help='混沌神经元个数')
    _m(g, '--chaos_kf', type=float, default=0.2)
    _m(g, '--chaos_kr', type=float, default=0.7)
    _m(g, '--chaos_alpha', type=float, default=1.0)
    _m(g, '--chaos_eps', type=float, default=0.04)
    _m(g, '--chaos_w_scale', type=float, default=0.5)
    _m(g, '--chaos_lr_scale', type=float, default=10.0,
       help='混沌动力学参数(kf/kr/alpha/eps)学习率倍率（相对 --lr_actor）')
    _m(g, '--n_actor_hidden', type=int, default=128)
    _m(g, '--n_critic_hidden', type=int, default=128)
    _m(g, '--activation', default='relu')

    g = ap.add_argument_group('贝叶斯策略网络')
    _m(g, '--bayes_prior_sigma', type=float, default=0.1)
    # ρ_init=-5 → 后验 σ=0.0067，几乎无扰动（且旧实现里 ρ 梯度被 ε·sigmoid(ρ) 压掉两个数量级）；
    # 放宽到 -3 → σ=0.049，配合 --rho_lr_scale 让变分后验真正参与学习。
    _m(g, '--bayes_rho_init', type=float, default=-3.0)
    _m(g, '--rho_lr_scale', type=float, default=20.0,
       help='变分参数 ρ 的学习率倍率（相对 --lr_actor）')
    _m(g, '--br_alpha', type=float, default=5e-4, help='贝叶斯正则（每参数平均 KL）权重')

    g = ap.add_argument_group('探索噪声（标准 TD3 高斯）')
    _m(g, '--explore_sigma', type=float, default=0.1,
       help='动作探索噪声标准差，以动作半宽为单位（原版 TD3 在 [-1,1] 动作空间取 0.1）')
    _m(g, '--log_freq', type=int, default=100, help='每多少次 actor 更新记录一次训练日志')

    g = ap.add_argument_group('诊断与输出')
    g.add_argument('--max_lag', type=int, default=3, help='滞后诊断最大阶数')
    g.add_argument('--perturb', default='',
                   help='鲁棒性扰动（只作用于评估段）：missing@0.1 / noise@0.03 / volatility@0.2 / '
                        'drift@1.1 / delay@2；缺失用因果前向填充')
    g.add_argument('--log_every_sec', type=float, default=30.0)
    _m(g, '--ablate_bayes', type=int, default=0,
       help='消融：1=把贝叶斯策略层换成确定性线性层（去掉后验与 KL）')
    _m(g, '--ablate_chaos', type=int, default=0,
       help='消融：1=把内生混沌编码器换成同维度 MLP 编码器')
    return ap


def model_kwargs(args) -> Dict:
    kw = {k: getattr(args, k) for k in MODEL_ARGS}
    bad = sorted(set(kw) - ACCEPTED_KWARGS)
    if bad:
        raise SystemExit(f"[错误] 模型不读取的超参：{bad}")
    return kw


def reward_overrides(args) -> Dict:
    return {k: getattr(args, k) for k in REWARD_ARGS}
