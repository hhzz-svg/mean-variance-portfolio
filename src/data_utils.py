# -*- coding: utf-8 -*-
"""
参数估计：从历史收益率估计期望收益向量 μ 与协方差矩阵 Σ。

包含两种协方差估计：
  1. 样本协方差 S —— 无偏但在维数较高 / 样本较少时病态（条件数大、可能近奇异）。
  2. Ledoit-Wolf 收缩估计 —— 统计学习（machine learning）方法，把样本协方差
     向结构化目标 F = m·I 收缩，自动选取最优收缩强度 δ*，得到良态（well-conditioned）
     的估计。这正是本项目“机器学习相关”的体现：用收缩/正则化改善估计量。

参考：Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional
covariance matrices", Journal of Multivariate Analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_returns(path: str) -> pd.DataFrame:
    """读取收益率 CSV（第一列 day 为索引）。"""
    return pd.read_csv(path, index_col="day")


def estimate_mu(returns: np.ndarray) -> np.ndarray:
    """样本期望收益（按列求均值）。返回形状 (n,)。"""
    return np.asarray(returns).mean(axis=0)


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """样本协方差矩阵 S（无偏估计，分母 T-1）。返回形状 (n, n)。"""
    return np.cov(np.asarray(returns), rowvar=False, bias=False)


def ledoit_wolf_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf 收缩协方差估计，目标为按比例的单位阵 F = m·I。

    估计量： Σ_LW = δ·(m·I) + (1-δ)·S
    其中 m = tr(S)/n 是平均方差，δ ∈ [0,1] 是数据驱动的最优收缩强度。

    采用 Ledoit-Wolf(2004) 的内积 <A,B> = tr(A Bᵀ)/n 推导的闭式 δ。

    返回
    ----
    (Sigma_lw, delta)
    """
    X = np.asarray(returns, dtype=float)
    T, n = X.shape
    # 去均值（收缩公式基于去均值后的二阶矩）
    Xc = X - X.mean(axis=0, keepdims=True)
    # 用 1/T 的样本二阶矩（与 LW 推导一致）
    S = (Xc.T @ Xc) / T

    m = np.trace(S) / n                       # 收缩目标的标量：平均方差
    # d² = (1/n)·||S - m·I||_F²
    d2 = np.sum((S - m * np.eye(n)) ** 2) / n

    # b̄² = (1/T²)·Σ_t (1/n)·||x_t x_tᵀ - S||_F²
    #     用恒等式 ||x x' - S||_F² = ||x||⁴ - 2·xᵀS x + ||S||_F²  做向量化
    norm_x4 = (np.sum(Xc ** 2, axis=1)) ** 2          # 每个样本 ||x_t||⁴
    quad = np.einsum("ti,ij,tj->t", Xc, S, Xc)        # 每个样本 x_tᵀ S x_t
    S_fro2 = np.sum(S ** 2)
    per_t = norm_x4 - 2.0 * quad + S_fro2
    b2_bar = np.sum(per_t) / (n * T * T)

    b2 = min(b2_bar, d2)                      # 截断保证 δ ∈ [0,1]
    delta = 0.0 if d2 == 0 else b2 / d2

    Sigma_lw = delta * m * np.eye(n) + (1.0 - delta) * S
    return Sigma_lw, float(delta)


def annualize_mu(mu_daily: np.ndarray) -> np.ndarray:
    """日频期望收益年化（× 交易日数）。"""
    return np.asarray(mu_daily) * TRADING_DAYS


def annualize_cov(cov_daily: np.ndarray) -> np.ndarray:
    """日频协方差年化（× 交易日数）。"""
    return np.asarray(cov_daily) * TRADING_DAYS


def condition_number(matrix: np.ndarray) -> float:
    """对称正定矩阵的条件数 = λ_max / λ_min（用于对比样本估计与收缩估计的良态性）。"""
    eig = np.linalg.eigvalsh(matrix)
    lo = eig.min()
    return float(eig.max() / lo) if lo > 0 else float("inf")


def condition_vs_window(returns: np.ndarray, windows):
    """计算条件数随估计窗口长度 T 的变化（样本协方差 vs Ledoit-Wolf 收缩）。

    用于展示“高维/短样本时收缩估计才真正重要”这一统计学习结论。
    返回 (cond_sample_list, cond_lw_list, delta_list)。
    """
    cond_s, cond_l, deltas = [], [], []
    for T in windows:
        Rw = np.asarray(returns)[:T]
        S = sample_covariance(Rw)
        Slw, d = ledoit_wolf_shrinkage(Rw)
        cond_s.append(condition_number(S))
        cond_l.append(condition_number(Slw))
        deltas.append(d)
    return cond_s, cond_l, deltas
