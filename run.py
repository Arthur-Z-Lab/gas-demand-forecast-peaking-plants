"""CCNN-BTD3 多步日负荷预测：训练（历史数据驱动，不与物理系统交互）→ 验证早停 → 区间校准 → 测试评估。"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - 仅防御性
        pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.paper.PAPER import PAPER, make_reward, set_seed
from src.artifacts import append_row, check_resume, summarize, write_json, write_predictions
from config import (DEFAULT_DELTA_HIGH, DEFAULT_DELTA_LOW, apply_best_config,
                    build_parser, model_kwargs, reward_overrides)
from src.data_loader import describe_series, load_series, prepare_folds
from src.efficiency import efficiency_report, reset_memory
from src.env import ForecastEnv
from src.evaluation import evaluate, fit_conformal, val_score
from src.figures import plot_all
from src.linear_prior import segment_priors, select_blend_weight
from src.perturbation import apply_perturbation, parse_perturbation
from src.reporting import fmt, hr, pct, print_block, print_result
from src.train import build_agent, train


def _rel(path) -> str:
    """用于协议指纹的数据标识：尽量用相对路径，避免同名文件互相覆盖记录。"""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def run_fold(args, raw, data_path, col, seg, fold_idx, n_folds, seeds, outdir, mkw):
    """单折全流程：训练（验证段早停 + 回滚）→ 测试段评估（含贝叶斯预测区间）→ 出图与落盘。

    `--perturb` 时对评估段施加鲁棒性扰动：输入侧扰动（缺失/噪声/波动）只污染模型可见序列，
    评估真值仍取原始观测；概念漂移同时改变真值；通信延迟由 env.delay 实现。
    """
    pname, pval = parse_perturbation(args.perturb)
    delay = int(round(pval)) if pname == 'delay' else 0
    scale, offset, z0 = seg['hi'] - seg['lo'], seg['lo'], seg['zero_level']
    info = describe_series(seg)


    rov = dict(reward_overrides(args), zero_level=z0)
    cfg = make_reward(args.horizon, rov)

    min_start = args.seq_len
    # 动作界：残差模式训练段经验界 ±0.80（见 config.DEFAULT_DELTA_*）
    a_low, a_high = DEFAULT_DELTA_LOW, DEFAULT_DELTA_HIGH
    p_max = args.p_max

    if args.perturb:
        seg = apply_perturbation(seg, args.perturb, np.random.default_rng(1000 * fold_idx + 7))
        seg['val_perturbed'] = seg['val']
        seg['test_perturbed'] = seg['test']

    env_tr = ForecastEnv(
        seg['train'],
        args.seq_len,
        cfg,
        p_max=p_max,
        stop_survival=args.stop_survival,
        off_threshold=args.off_threshold,
        gate_temp=args.gate_temp,
        gate_mode=args.gate_mode,
    )
    surv = env_tr._surv if args.stop_survival else None
    env_va = ForecastEnv(
        seg.get('val_perturbed', seg['val']),
        args.seq_len,
        cfg,
        min_start=min_start,
        p_max=p_max,
        surv_table=surv,
        off_threshold=args.off_threshold,
        gate_temp=args.gate_temp,
        gate_mode=args.gate_mode,
        truth=seg.get('truth_val', None),
        delay=delay,
    )
    env_te = ForecastEnv(
        seg.get('test_perturbed', seg['test']),
        args.seq_len,
        cfg,
        min_start=min_start,
        p_max=p_max,
        surv_table=surv,
        off_threshold=args.off_threshold,
        gate_temp=args.gate_temp,
        gate_mode=args.gate_mode,
        truth=seg.get('truth_test', None),
        delay=delay,
    )

    q = info['train_dy_q']
    seg_desc = " │ ".join(
        f"{k} 零值 {pct(info[k]['zero'])} 最长停机 {info[k]['max_zero_run']} 日 启停 {info[k]['switch_per_100']:.1f}/百日"
        for k in ('train', 'val', 'test'))
    fold_desc = f"（第 {fold_idx}/{n_folds} 折 walk-forward）" if n_folds > 1 else ""
    print_block(f"CBR-TD3 多步负荷预测{fold_desc}", [
        ("数据", f"{data_path.name}::{col} | N={len(raw)} | train/val/test = "
                 f"{seg['n_train']}/{seg['n_val']}/{seg['n_test']}"
                 f"（切分点 {seg['span'][1]}/{seg['span'][2]}/{seg['span'][3]}，时间顺序，min-max 仅用训练段）"),
        ("负荷特征", seg_desc),
        ("日间变化", f"训练段 |Δy| 分位数 q50/q90/q95/q99 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}/{q[3]:.3f}（标幺）"),
        ("样本组", f"L={args.seq_len} 日 → H={args.horizon} 日 | 训练集合 {env_tr.n_episodes} 个起报日（随机抽取，"
                   f"组内按步顺序）| 验证 {env_va.n_episodes} / 测试 {env_te.n_episodes}（时间顺序）"),
        ("训练预算", f"{args.total_episodes:,} episode，每轮并行 {args.n_parallel}，UTD={args.utd}，"
                     f"batch={args.batch_size}，每 {args.val_every:,} episode 验证"),
        ("决策过程", f"动作 u=[Δ,p_off]，Δ∈[{a_low:.3f},{a_high:.3f}]；正负荷分支相对"
                     f"起报锚点与上一步预测各半；p_off≥{args.off_threshold:.2f} 判停；γ={args.gamma}"),
        ("动作界", f"固定界 Δ∈[{a_low:+.3f},{a_high:+.3f}]（训练段经验界）"),
        ("停机先验", "训练段 P(次日仍停|已停 c 天) 仅作为状态特征，预测状态逐步更新 run；验证/测试复用训练表"
                     if args.stop_survival else "关闭经验持续概率特征（仍保留 run-length）"),
        ("奖励 LCAR", f"点损失 {cfg.point_scale:g}×e²（平方误差）"
                      f" β_Δ={cfg.beta_change} β_p={cfg.beta_peak} λ_Δ={cfg.lam_delta} "
                      f"λ_dir={cfg.lam_dir} λ_s={cfg.lam_state} λ_g={cfg.lam_growth}"),
        ("网络", f"CCNN({args.n_chaos}) + 全 Bayesian 策略读出（Δ/off）；双 CCNN Critic"
                 f" | load-loss {args.anchor_weight} | off-BCE {args.off_loss_weight}×pos {args.off_pos_weight}×switch {args.off_switch_weight}"),
        ("预测区间", f"{int(args.pi_level * 100)}%：{args.mc_samples} 组贝叶斯策略后验采样"
                     + ("（另加验证段共形校准，同时输出 Cal_* 指标）"
                        if args.pi_method == 'conformal' else "（Bayes_* 指标）")),
        ("种子/输出", f"{seeds} → {outdir}"),
    ])
    if info['test']['max_zero_run'] > max(info['train']['max_zero_run'], args.seq_len):
        print(f"  ⚠️ 测试段最长连续停机 {info['test']['max_zero_run']} 日，超过训练段 "
              f"{info['train']['max_zero_run']} 日：属于训练中未出现的工况，请结合启停分组指标解读。")
    config = dict(model=mkw, reward=cfg.to_dict(), protocol=dict(
        data=f"{_rel(data_path)}#{data_path.stat().st_size}", col=col,
        split=[args.train_ratio, args.val_ratio], n_folds=n_folds, fold=fold_idx,
        max_lag=args.max_lag, min_start=min_start, L=args.seq_len,
        H=args.horizon, total_episodes=args.total_episodes, n_parallel=args.n_parallel, utd=args.utd,
        warmup_episodes=args.warmup_episodes, early_stop=args.early_stop, val_every=args.val_every,
        patience=args.patience, min_episodes=args.min_episodes,
        stop_survival=args.stop_survival, off_threshold=args.off_threshold, val_metric=args.val_metric,
        a_low=a_low, a_high=a_high, p_max=p_max,
        pi_level=args.pi_level, mc_samples=args.mc_samples, gate_temp=args.gate_temp,
        # 复算协议：记录区间口径/是否用线性先验/奖励类型/验证准则，供 retest.py 精确复现
        pi_method=args.pi_method, use_prior=int(args.use_prior),
        reward_type=cfg.reward_type, action_dim=2, perturb=str(args.perturb or 'none')))
    config_json = json.dumps(config, sort_keys=True, ensure_ascii=False, default=float)
    tag = f"{data_path.stem}_L{args.seq_len}_H{args.horizon}" + (f"_f{fold_idx}" if n_folds > 1 else "")
    trials = outdir / f"trials_{tag}.csv"
    done = check_resume(trials, config_json)

    for seed in seeds:
        if seed in done:
            print(f"  [跳过] seed {seed} 已完成")
            continue
        hr()
        print(f"  ▶ seed {seed}")
        set_seed(seed)
        reset_memory()
        agent = build_agent(env_tr, PAPER, mkw, a_low=a_low, a_high=a_high)
        lyap0 = agent.lyapunov()['lyap_actor']
        t0 = time.time()
        log = train(agent, env_tr, total_episodes=args.total_episodes, n_parallel=args.n_parallel,
                    utd=args.utd, warmup_episodes=args.warmup_episodes, rng=np.random.default_rng(seed),
                    val_fn=(lambda ag: val_score(env_va, ag, args.val_metric)) if args.early_stop else None,
                    val_every=args.val_every, patience=args.patience, min_episodes=args.min_episodes,
                    log_every_sec=args.log_every_sec)
        log['n_parallel'] = args.n_parallel
        t_train = time.time() - t0

        # 可选：线性先验门控混合（--use_prior 1；消融"线性兜底对精度是否有用"）
        bw, priors = None, None
        if args.use_prior:
            priors = segment_priors(seg, args.seq_len, args.horizon,
                                    alpha=args.ridge_alpha, n_lag=args.ridge_lags)
            blend = select_blend_weight(env_va, agent, priors['val'], metric=args.blend_metric)
            bw = float(blend['weight'])
            print(f"    [线性先验混合] 验证段最优 w={bw:.2f}（验证 MAE {blend['mae']:.6f}）")

        # 可选：共形区间（--pi_method conformal）；默认用贝叶斯后验采样区间
        cq, cq_map = None, None
        if args.pi_method == 'conformal':
            from src.evaluation import PI_LEVELS
            cq, cq_map = fit_conformal(env_va, agent, args.mc_samples, args.pi_level,
                                       blend_w=bw, blend_prior=(priors['val'] if priors else None),
                                       levels=PI_LEVELS)

        # 点预测 = 策略后验均值 rollout（--use_prior 1 时叠加线性先验混合）
        m, ro = evaluate(env_te, agent, cfg, scale=scale, offset=offset, max_lag=args.max_lag,
                         n_samples=args.mc_samples, level=args.pi_level,
                         blend_w=bw, blend_prior=(priors['test'] if priors else None),
                         conformal_q=cq, pi_method=args.pi_method, conformal_qmap=cq_map,
                         ramp_limit=float(np.quantile(np.abs(np.diff(seg['raw_train'])), 0.95)))
        m.update(agent.lyapunov(), lyap_actor_init=lyap0,
                 val_MAE_norm=log['best_val'] if args.early_stop else val_score(env_va, agent, args.val_metric),
                 best_episode=log['best_episode'], episodes_run=log['episodes_run'],
                 updates=log['updates'], transitions=log['transitions'], train_time_s=round(t_train, 1),
                 off_threshold_used=float(env_te.off_threshold),
                 use_prior=int(args.use_prior), blend_w=(bw if bw is not None else float('nan')),
                 perturb=str(args.perturb or 'none'),
                 reward_extras=int(args.reward_extras), val_metric=str(args.val_metric),
                 **efficiency_report(agent, env_te))

        stem = f"{tag}_s{seed}"
        write_predictions(outdir / f"preds_{stem}.csv", ro, scale, offset, seg['offset_test'])
        # 验证段预测一并落盘：后续换指标/换口径/重算区间都无需重新训练
        ro_va = env_va.rollout(agent.act_batch, env_va.starts)
        write_predictions(outdir / f"valpreds_{stem}.csv", ro_va, scale, offset, seg['offset_val'])
        agent.save_model(outdir / f"actor_{stem}.pt")
        agent.save_logs(outdir / f"logs_{stem}.pkl")
        if not args.no_figures:
            fig_dir = plot_all(outdir / f"preds_{stem}.csv", log, agent.logs, outdir / 'figures', stem, args.horizon)
            print(f"    图片已保存：{fig_dir}")
        append_row(trials, dict(launch=datetime.now().strftime('%Y%m%d_%H%M%S'), seed=seed,
                                fold=fold_idx, n_folds=n_folds, config=config_json, **m))
        print_result(f"seed {seed}", m, args.pi_level)
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.read_csv(trials)
    s = summarize(df, outdir / f"summary_{tag}.csv")
    write_json(outdir / f"config_{tag}.json", dict(config, seeds=seeds, denorm_scale=scale, denorm_offset=offset))
    hr("═")
    print(f"  汇总（{len(df)} 个种子，均值 ± 标准差）")
    for k, nd, sci in (('MAE', 3, True), ('RMSE', 3, True), ('R2', 4, False),
                       ('MAE_h1', 3, True), (f'MAE_h{args.horizon}', 3, True),
                       ('DirAcc', 4, False), ('OnOffAcc', 4, False), ('OffRecall', 4, False),
                       ('Acc_ON_OFF', 4, False), ('Acc_OFF_ON', 4, False),
                       ('ShutdownRate', 4, False), ('MAE_on', 3, True), ('MAE_off', 3, True),
                       ('ErrShare_off', 4, False),
                       ('Bayes_PICP', 4, False), ('CRPS', 3, True)):
        if k not in s.index:
            continue
        line = f"    {k:<14} {fmt(s.loc[k, 'mean'], nd, sci)} ± {fmt(s.loc[k, 'std'], nd, sci)}"
        print(line)
    print(f"  文件：{trials.name} / summary_{tag}.csv / preds_*.csv / actor_*.pt / logs_*.pkl / figures/*.png")
    hr("═")
    return tag, trials


def summarize_folds(records, outdir, tag_root):
    """跨折汇总：把每折 summary 的关键指标再按折取均值/标准差，暴露折间波动。"""
    keys = ('R2', 'MAE_norm', 'MAE_on', 'MAE_off', 'ErrShare_off', 'ShutdownRate',
            'DirAcc', 'OnOffAcc', 'OffRecall', 'Acc_ON_OFF', 'Acc_OFF_ON', 'Bayes_PICP', 'CRPS')
    rows = []
    for tag, _ in records:
        s = pd.read_csv(outdir / f"summary_{tag}.csv", index_col=0)
        for k in keys:
            if k in s.index:
                sd = s.loc[k, 'std']
                rows.append(dict(tag=tag, metric=k, mean=float(s.loc[k, 'mean']),
                                 std=float(sd) if pd.notna(sd) else float('nan')))
    if not rows:
        return
    df = pd.DataFrame(rows)
    piv = df.pivot(index='metric', columns='tag', values='mean').reindex(list(keys)).dropna(how='all')
    piv['folds_mean'] = piv.mean(axis=1)
    piv['folds_std'] = piv.std(axis=1, ddof=1) if len(records) > 1 else float('nan')
    out = outdir / f"summary_folds_{tag_root}.csv"
    piv.to_csv(out)
    hr("═")
    print(f"  跨折汇总（{len(records)} 折；看折间均值与标准差判断稳定性）")
    hr("═")
    for k, r in piv.iterrows():
        print(f"    {k:<14} 折间均值 {fmt(r['folds_mean'], 4)} ± {fmt(r['folds_std'], 4)}")
    print(f"  文件：{out.name}")
    hr("═")


def main():
    args = build_parser().parse_args()
    apply_best_config(args)
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mkw = model_kwargs(args)
    data_path = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    raw, col = load_series(data_path, args.col)
    folds = prepare_folds(raw, args.n_folds, args.train_ratio, args.val_ratio)
    only_fold = int(getattr(args, 'only_fold', 0) or 0)
    if only_fold and not (1 <= only_fold <= len(folds)):
        raise SystemExit(f"[错误] --only_fold 需在 1..{len(folds)} 之间，实得 {only_fold}")
    records = []

    for fi, seg in enumerate(folds, start=1):
        if only_fold and fi != only_fold:
            continue
        records.append(run_fold(args, raw, data_path, col, seg, fi, len(folds),
                                seeds, outdir, mkw))

    # --only_fold 时不做跨折汇总：records 只有一折，避免用单折结果覆盖完整运行的 summary_folds_*.csv
    if len(folds) > 1 and not only_fold:
        summarize_folds(records, outdir, f"{data_path.stem}_L{args.seq_len}_H{args.horizon}")


if __name__ == '__main__':
    main()
