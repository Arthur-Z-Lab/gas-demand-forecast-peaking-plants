"""鲁棒性实验的工业场景扰动（审稿人要求：数据缺失、通信延迟、运行策略变更、概念漂移、波动）。

语义约定（关键）：
  * **输入侧扰动**（缺失 / 噪声 / 波动）：模型看到的历史被污染，但**评估真值仍是原始观测**
    —— 环境用 `series=扰动后输入`、`truth=原始真值` 构造；
  * **工况侧扰动**（概念漂移 / 运行策略变更）：真值本身变化，输入与真值同时改变；
  * **通信延迟**：由环境的 `delay` 参数实现（可见历史整体滞后 δ 天，真值不变）。

训练段一律不扰动，扰动只作用于评估段；归一化尺度仍取原训练段。

支持的 `--perturb` 取值：
  missing@0.10      缺失 10% 观测 → 因果前向填充（不填 0）
  noise@0.03        3% 量程的高斯量测噪声
  volatility@0.20   20% 量程的 AR(1) 高频波动（可再生/调度波动近似）
  drift@1.10        概念漂移/策略变更：评估段负荷整体 ×1.10
  delay@2           通信延迟 2 天
"""
from typing import Dict, Tuple

import numpy as np

__all__ = ["apply_perturbation", "parse_perturbation", "INPUT_SIDE", "SHIFT_SIDE"]

INPUT_SIDE = ("missing", "noise", "volatility")     # 只污染输入，真值不变
SHIFT_SIDE = ("drift",)                             # 真值同变


def parse_perturbation(spec: str) -> Tuple[str, float]:
    """'missing@0.10' → ('missing', 0.10)；空串 → ('', 0.0)"""
    if not spec:
        return "", 0.0
    name, _, val = str(spec).partition("@")
    return name.strip(), (float(val) if val else 0.0)


def apply_perturbation(seg: Dict, spec: str, rng: np.random.Generator) -> Dict:
    """按 spec 扰动评估段；返回的新 seg 中 `val/test` 是输入序列，`truth_val/truth_test` 是真值序列。

    未扰动时 `truth_*` 与 `val/test` 相同，调用方可统一按 truth_* 传参。
    """
    name, val = parse_perturbation(spec)
    out = dict(seg)
    out["truth_val"] = np.asarray(seg["val"], float).copy()
    out["truth_test"] = np.asarray(seg["test"], float).copy()
    out["perturb"] = spec or "none"
    if not name:
        return out
    # 注意：seg['val']/seg['test'] 是**归一化**序列（训练段 min-max），
    # 因此扰动幅度直接以 val 为单位（如 noise@0.05 = 5% 训练段量程），不要再乘原始量程。

    def corrupt(s: np.ndarray) -> np.ndarray:
        if name == "missing":
            s2 = s.copy()
            miss = rng.random(len(s2)) < float(val)
            s2[miss] = np.nan
            idx = np.where(~np.isnan(s2), np.arange(len(s2)), 0)
            np.maximum.accumulate(idx, out=idx)        # 因果前向填充
            filled = s2[idx]
            lead = int(np.argmax(~np.isnan(s2))) if np.any(~np.isnan(s2)) else 0
            if lead:                                   # 起始缺失段用首个有效观测回填（保持因果）
                filled[:lead] = s2[lead]
            return filled
        if name == "noise":
            return np.clip(s + rng.normal(0.0, float(val), size=len(s)), 0.0, None)
        if name == "volatility":
            phi, amp = 0.7, float(val)
            e = np.zeros(len(s))
            for t in range(1, len(s)):
                e[t] = phi * e[t - 1] + rng.normal(0.0, amp * np.sqrt(1 - phi ** 2))
            return np.clip(s + e, 0.0, None)
        if name == "drift":
            v = float(val)
            return np.clip(s * v if abs(v - 1.0) <= 0.5 else s + v, 0.0, None)
        return s

    if name == "delay":
        return out                                     # 延迟由 env.delay 处理（真值不变）
    for key in ("val", "test"):
        out[key] = corrupt(np.asarray(seg[key], float))
        if name in SHIFT_SIDE:                         # 概念漂移：真值一起变
            out[f"truth_{key}"] = out[key]
    return out
