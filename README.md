# CBR-TD3: chaos-aware Bayesian reinforcement learning framework with TD3 optimization

Code accompanying the paper:

> **Multi-Step Natural Gas Demand Forecasting for Gas-Fired Peaking Plants: Chaotic State Encoding and
> Transition-Aware Reinforcement Learning** (submitted to *Engineering Applications of Artificial Intelligence*)

Multi-step daily natural gas load forecasting for gas-fired peaking plants is formulated as a finite-horizon
sequential decision process. Each action contains a continuous load adjustment and an OFF probability, and
every prediction updates the recursive state used by the next forecasting step. The framework combines an
endogenous chaotic state encoder, a regime-aware Bayesian actor-critic trained with TD3, a load-change-aware
reward (LCAR), and a validation-selected ridge-regression output fusion.

## 1. Method overview

![CBR-TD3 framework](docs/figures/fig3_framework.png)

*Overall framework: data and MDP formulation, Bayesian chaotic TD3 training, and inference with calibrated
prediction intervals. The numbered markers correspond to the steps of Algorithm 1 in the paper.*

The three components are summarized below.

## 2. Repository layout

| Path | Content |
| --- | --- |
| `model/paper/PAPER.py` | Chaotic encoder, Bayesian actor-critic (TD3), semi-Markov ON/OFF heads, posterior sampling |
| `model/paper/reward.py` | Load-change-aware reward (LCAR) |
| `src/env.py` | Sequential forecasting environment (rollout, state transition, hurdle gate) |
| `src/data_loader.py` | Chronological train/validation/test split, expanding-window walk-forward folds |
| `src/train.py` | Training loop, Adam optimisation, parallel episodes, update-to-data ratio |
| `src/linear_prior.py` | Ridge-regression prior and validation-selected output fusion |
| `src/evaluation.py`, `src/metrics.py` | Point, interval and operating-state metrics, split conformal calibration, DM and block-bootstrap tests |
| `src/artifacts.py`, `src/figures.py`, `src/plot_style.py` | Per-seed metric tables, prediction files, training and forecast figures |
| `run.py`, `config.py` | Command-line entry point and the single source of truth for hyperparameters |
| `tools/dm_test.py` | Paired significance tests for two runs (Diebold-Mariano, moving-block bootstrap, Wilcoxon) |
| `scripts/run_protocol.sh` | The 5 horizons x 5 folds x 5 seeds protocol used in the paper |
| `data/HDQS.xlsx` | Representative samples of the primary series (see Section 3) |




