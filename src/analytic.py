# -*- coding: utf-8 -*-
"""
解析法求解器：拉格朗日 / KKT 闭式解。

求解（无 w≥0 约束、允许卖空的）均值-方差问题：

    min_w  ½ wᵀΣw      s.t.  μᵀw = r,  𝟙ᵀw = 1

拉格朗日一阶条件给出  Σw = λμ + γ𝟙  ⇒  w = Σ⁻¹(λμ + γ𝟙)，
代入两个约束得到关于 (λ, γ) 的 2×2 线性方程组。定义标量

    A = 𝟙ᵀΣ⁻¹μ,  B = μᵀΣ⁻¹μ,  C = 𝟙ᵀΣ⁻¹𝟙,  D = BC − A²

则任意目标收益 r 对应的最优组合方差有闭式

    σ²(r) = (C·r² − 2A·r + B) / D          ——  有效前沿（抛物线/双曲线）

本模块用 **Cholesky 分解 + 前代/回代** 来求解 Σ⁻¹ 作用在向量上的结果，
而非显式求逆——这是数值线性代数中对对称正定系统的标准做法（更稳定、更省）。
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# 矩阵算法基石：对称正定系统的 Cholesky 求解（手写前代/回代）
# --------------------------------------------------------------------------
def _forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """解下三角系统 L y = b。"""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def _back_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """解上三角系统 U x = y。"""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def chol_solve(Sigma: np.ndarray, b: np.ndarray) -> np.ndarray:
    """对对称正定 Σ，求解 Σ x = b（即 x = Σ⁻¹ b），经由 Σ = L Lᵀ。"""
    L = np.linalg.cholesky(Sigma)          # Σ = L Lᵀ，L 为下三角
    y = _forward_substitution(L, b)        # L y = b
    x = _back_substitution(L.T, y)         # Lᵀ x = y
    return x


# --------------------------------------------------------------------------
# 有效前沿的核心标量与组合
# --------------------------------------------------------------------------
def frontier_scalars(mu: np.ndarray, Sigma: np.ndarray) -> dict:
    """计算 A, B, C, D 及 Σ⁻¹μ、Σ⁻¹𝟙（后续所有闭式解的公共部分）。"""
    n = len(mu)
    ones = np.ones(n)
    Sinv_mu = chol_solve(Sigma, mu)
    Sinv_1 = chol_solve(Sigma, ones)
    A = float(ones @ Sinv_mu)              # 1ᵀΣ⁻¹μ
    B = float(mu @ Sinv_mu)                # μᵀΣ⁻¹μ
    C = float(ones @ Sinv_1)               # 1ᵀΣ⁻¹1
    D = B * C - A * A
    return dict(A=A, B=B, C=C, D=D, Sinv_mu=Sinv_mu, Sinv_1=Sinv_1)


def weight_for_return(mu: np.ndarray, Sigma: np.ndarray, r: float,
                      sc: dict | None = None) -> np.ndarray:
    """给定目标收益 r，返回有效前沿上对应的最优权重（允许卖空）。"""
    if sc is None:
        sc = frontier_scalars(mu, Sigma)
    A, B, C, D = sc["A"], sc["B"], sc["C"], sc["D"]
    lam = (C * r - A) / D                   # 对应 μ 的拉格朗日乘子
    gam = (B - A * r) / D                   # 对应 𝟙 的拉格朗日乘子
    return lam * sc["Sinv_mu"] + gam * sc["Sinv_1"]


def frontier_variance(r, sc: dict) -> np.ndarray:
    """有效前沿闭式方差 σ²(r) = (C r² − 2A r + B)/D，可对数组 r 矢量化。"""
    A, B, C, D = sc["A"], sc["B"], sc["C"], sc["D"]
    r = np.asarray(r, dtype=float)
    return (C * r ** 2 - 2 * A * r + B) / D


def efficient_frontier(mu: np.ndarray, Sigma: np.ndarray, n_points: int = 80):
    """生成有效前沿上的一系列点。

    返回 (rets, vols, weights)：目标收益数组、对应波动率、对应权重矩阵 (n_points, n)。
    收益扫描范围取在全局最小方差组合收益之上（前沿的“有效”上半支）。
    """
    sc = frontier_scalars(mu, Sigma)
    A, C = sc["A"], sc["C"]
    r_gmv = A / C
    r_max = mu.max()
    rets = np.linspace(r_gmv, r_gmv + 1.15 * (r_max - r_gmv), n_points)
    vols = np.sqrt(frontier_variance(rets, sc))
    weights = np.array([weight_for_return(mu, Sigma, r, sc) for r in rets])
    return rets, vols, weights


def gmv_portfolio(mu: np.ndarray, Sigma: np.ndarray) -> dict:
    """全局最小方差组合 (GMV)：w = Σ⁻¹𝟙 / (𝟙ᵀΣ⁻¹𝟙)。"""
    sc = frontier_scalars(mu, Sigma)
    A, C = sc["A"], sc["C"]
    w = sc["Sinv_1"] / C
    return dict(weights=w, ret=A / C, var=1.0 / C, vol=float(np.sqrt(1.0 / C)))


def tangency_portfolio(mu: np.ndarray, Sigma: np.ndarray, rf: float) -> dict:
    """切点（最大夏普）组合：w ∝ Σ⁻¹(μ − r_f·𝟙)，再归一化使和为 1。"""
    n = len(mu)
    ones = np.ones(n)
    excess = mu - rf * ones
    z = chol_solve(Sigma, excess)
    w = z / (ones @ z)
    ret = float(mu @ w)
    var = float(w @ Sigma @ w)
    vol = float(np.sqrt(var))
    sharpe = (ret - rf) / vol
    return dict(weights=w, ret=ret, var=var, vol=vol, sharpe=sharpe)
