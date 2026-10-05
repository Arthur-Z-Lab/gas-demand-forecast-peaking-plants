"""可部署性指标：参数量、权重文件大小、单起报日 H 步推理延迟、训练峰值显存 / 进程内存。"""
import os
import tempfile
import time

import numpy as np
import torch

__all__ = ["reset_memory", "efficiency_report"]


def reset_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20
    except ImportError:
        try:                      # Linux 无 psutil 时读 /proc（与 psutil 等价）
            with open("/proc/self/status", "r") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return float("nan")


def efficiency_report(agent, env, n_runs: int = 200) -> dict:
    gpu_peak = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else float("nan")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    torch.save(agent.actor.state_dict(), path)
    size = os.path.getsize(path) / 2 ** 20
    os.unlink(path)
    one = env.starts[:1]
    for _ in range(10):
        env.rollout(agent.act_batch, one)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        env.rollout(agent.act_batch, one)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / n_runs * 1000
    n = lambda m: int(sum(p.numel() for p in m.parameters()))
    return dict(params_actor=n(agent.actor), params_critic=n(agent.critic),
                actor_file_mb=round(size, 4), infer_ms_per_origin=round(lat, 3),
                gpu_peak_mb=round(gpu_peak, 1), rss_mb=round(_rss_mb(), 1))
