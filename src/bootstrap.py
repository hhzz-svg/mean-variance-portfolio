# -*- coding: utf-8 -*-
"""
Circular block bootstrap：对样本外日收益做带时序依赖的显著性推断。

动机
----
回测给出的“策略 A 夏普 0.45 > 策略 B 夏普 0.20”只是**点估计**，一条 9 年的
收益路径只是一次随机抽样——差距是真实优势还是运气？bootstrap 用重采样近似
夏普比率的抽样分布，给出置信区间与 p 值。

两个关键设计
------------
1. **块抽样（block）而非逐日抽样**：日收益有波动聚集等时序依赖，逐日独立
   重采会破坏依赖结构、低估方差。按长度为 block 的连续块抽取（环形拼接，
   Politis-Romano 风格），块内依赖被保留。块长取再平衡周期 21 天。
2. **联合重采（joint resampling）**：所有策略的收益矩阵 (T, m) 用**同一组**
   块索引切片。这保留了策略间的截面相关——检验“ΔSharpe = 夏普(A) − 夏普(B)”
   时，两者在同一 resample 内同涨同跌的部分相互抵消，检验才有功效；
   若各自独立重采，Δ 的方差会被严重高估。

年化夏普在每个 resample 内**整段重算**：
    sharpe_b = (mean(r⁽ᵇ⁾)·252 − rf) / (std(r⁽ᵇ⁾, ddof=1)·√252)
（不能分别 bootstrap 均值和波动再组合——两者在同一 resample 内相关；
ddof=1 与 metrics.annualized_stats 口径严格一致。）

p 值用 (count+1)/(B+1) 修正，避免报出 p=0（B=2000 时分辨率下限 ≈ 0.001）。
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def circular_block_indices(T: int, block: int, rng) -> np.ndarray:
    """生成一条长 T 的环形块索引：每块起点 ~ U[0,T)，连续取 block 天（模 T 回绕），
    共 ⌈T/block⌉ 块拼接后截断到 T。"""
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % T
    return idx.ravel()[:T]


def _annualized_sharpe(mat: np.ndarray, rf_annual: float) -> np.ndarray:
    """(T, m) 日收益矩阵 → (m,) 年化夏普。口径同 metrics.annualized_stats。"""
    ann_ret = mat.mean(axis=0) * TRADING_DAYS
    ann_vol = mat.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
    return (ann_ret - rf_annual) / ann_vol


def joint_sharpe_bootstrap(daily_matrix: np.ndarray, labels: list[str],
                           rf_annual: float, benchmark: str,
                           block: int = 21, B: int = 2000,
                           seed: int = 2024, alpha: float = 0.05) -> dict:
    """对联合日收益矩阵做 circular block bootstrap，返回夏普 CI 与 vs 基准的检验。

    参数
    ----
    daily_matrix : (T, m) 样本外日收益，第 j 列对应 labels[j]，全部策略同一时间轴。
    benchmark    : ΔSharpe 检验的基准策略名（须在 labels 中）。

    返回
    ----
    dict：
      sharpe_point   (m,)   全序列点估计；
      sharpe_ci      (m,2)  percentile 法 (1−α) 置信区间；
      delta_vs_benchmark {label: {delta_point, ci, p_value}}  联合重采下的 ΔSharpe；
      sharpe_samples (B,m)  bootstrap 样本（绘图用）；
      config         参数记录。
    """
    Rm = np.asarray(daily_matrix, dtype=float)
    T, m = Rm.shape
    assert len(labels) == m, "labels 与矩阵列数不一致"
    assert benchmark in labels, f"基准 {benchmark} 不在 labels 中"

    rng = np.random.default_rng(seed)
    point = _annualized_sharpe(Rm, rf_annual)

    samples = np.empty((B, m))
    for b in range(B):                       # 逐 b 切片，不物化 (B,T,m) 大数组
        idx = circular_block_indices(T, block, rng)
        samples[b] = _annualized_sharpe(Rm[idx], rf_annual)

    lo = np.percentile(samples, 100.0 * alpha / 2.0, axis=0)
    hi = np.percentile(samples, 100.0 * (1.0 - alpha / 2.0), axis=0)

    j = labels.index(benchmark)
    delta = {}
    for i, lab in enumerate(labels):
        d = samples[:, i] - samples[:, j]    # 同一 resample 内作差 → 截面相关被保留
        n_le = int(np.sum(d <= 0.0))
        n_ge = int(np.sum(d >= 0.0))
        p = 2.0 * min(n_le + 1, n_ge + 1) / (B + 1)
        delta[lab] = {
            "delta_point": float(point[i] - point[j]),
            "ci": [float(np.percentile(d, 100.0 * alpha / 2.0)),
                   float(np.percentile(d, 100.0 * (1.0 - alpha / 2.0)))],
            "p_value": float(min(p, 1.0)),
        }

    return {
        "labels": list(labels),
        "sharpe_point": point,
        "sharpe_ci": np.column_stack([lo, hi]),
        "delta_vs_benchmark": delta,
        "sharpe_samples": samples,
        "config": {"block": block, "B": B, "seed": seed,
                   "alpha": alpha, "benchmark": benchmark, "T": T},
    }
