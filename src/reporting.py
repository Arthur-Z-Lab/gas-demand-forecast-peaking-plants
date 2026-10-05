import numpy as np

__all__ = ["hr", "fmt", "pct", "print_block", "print_result"]


def hr(ch="─", w=110):
    print(ch * w)


def fmt(x, nd=4, sci=False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v:.{nd}e}" if sci else f"{v:.{nd}f}"


def pct(x, nd=1):
    s = fmt(None if x is None else 100 * float(x), nd)
    return s if s == "—" else s + "%"


def print_block(title, lines):
    hr("═")
    print(f"  {title}")
    hr("═")
    for k, v in lines:
        print(f"  {k:<12}: {v}")
    hr("═")


def print_result(tag, m, level):
    """只输出论文/实验评估指标；训练内部量（梯度、Q、Lyapunov、参数量等）不打印。"""
    H = int(m['horizon'])
    e = lambda k: fmt(m.get(k), 3, True)
    hr()
    print(f"  {tag} │ 测试段 {int(m['n_origins'])} 个起报日 × {H} 日")
    hr()
    print(f"  整体精度      MAE {e('MAE')}  RMSE {e('RMSE')}  R² {fmt(m['R2'])}  CORR {fmt(m['CORR'])}"
          f"  Bias {e('Bias')} │ 标幺 MAE {fmt(m['MAE_norm'], 5)}")
    print("  分步 MAE      " + "  ".join(f"h{h}:{e(f'MAE_h{h}')}" for h in range(1, H + 1)))
    print("  分步 R²       " + "  ".join(f"h{h}:{fmt(m.get(f'R2_h{h}'), 3)}" for h in range(1, H + 1)))
    print(f"  变化预测      方向准确率 {pct(m.get('DirAcc'))} │ 增量 MAE {e('MAE_delta')}"
          f" │ 大变化日 MAE {e('MAE_bigchange')} │ 峰段 RMSE {e('RMSE_peak')}")
    print(f"  启停状态      准确率 {pct(m.get('OnOffAcc'))} │ 停机召回 {pct(m.get('OffRecall'))}"
          f" │ 开机召回 {pct(m.get('OnRecall'))} │ Brier {fmt(m.get('OffBrier'), 4)}"
          f" │ LogLoss {fmt(m.get('OffLogLoss'), 4)}")
    print(f"  状态转移      ON→ON {pct(m.get('Acc_ON_ON'))} │ ON→OFF {pct(m.get('Acc_ON_OFF'))}"
          f" │ OFF→OFF {pct(m.get('Acc_OFF_OFF'))} │ OFF→ON {pct(m.get('Acc_OFF_ON'))}"
          f" │ 切换日 MAE {e('MAE_switch')}")
    print(f"  分工况        停机占比 {pct(m.get('ShutdownRate'))} │ 开机段 MAE {e('MAE_on')}"
          f" │ 停机段 MAE {e('MAE_off')} │ 停机误差贡献 {pct(m.get('ErrShare_off'))}"
          f" │ 切换日 MAE {e('MAE_switch')}")
    print(f"  预测区间 {int(level * 100)}%  PICP {pct(m.get('PICP'))}"
          f"（h1 {pct(m.get('PICP_h1'))}, h{H} {pct(m.get(f'PICP_h{H}'))}）"
          f" │ PINAW {fmt(m.get('PINAW'), 4)} │ Winkler {e('Winkler')} │ CRPS {e('CRPS')}")
    print(f"  验证指标      best val MAE_norm {fmt(m.get('val_MAE_norm'), 5)}")
    hr()
