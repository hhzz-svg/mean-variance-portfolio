# -*- coding: utf-8 -*-
"""
数值法求解器：投影梯度下降（Projected Gradient Descent, PGD）。

当加入“不可卖空”约束 w ≥ 0 时，问题变为带不等式约束的二次规划，没有闭式解：

    min_w  ½ wᵀΣw − λ·μᵀw      s.t.  𝟙ᵀw = 1,  w ≥ 0

可行域是概率单纯形 Δ = { w : wᵢ ≥ 0, Σwᵢ = 1 }。
PGD 迭代：先沿负梯度走一步，再欧氏投影回单纯形

    w_{k+1} = Proj_Δ( w_k − η·∇f(w_k) ),     ∇f(w) = Σw − λμ

目标 f 凸（Σ 半正定），∇f 利普希茨常数 L = λ_max(Σ)，取步长 η ≤ 1/L 可保证收敛，
收敛速率为 O(1/k)。

单纯形投影用经典的“排序 + 阈值”算法（Held et al. 1974; Wang & Carreira-Perpiñán 2013），
其本身就是投影子问题 min_w ½‖w−v‖² s.t. w∈Δ 的 KKT 条件的解。
"""
from __future__ import annotations

import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """欧氏投影到概率单纯形 {w ≥ 0, Σw = 1}。

    解 min_w ½‖w − v‖²  s.t. wᵢ ≥ 0, Σwᵢ = 1。
    KKT 给出 wᵢ = max(vᵢ − θ, 0)，θ 由 Σ max(vᵢ−θ,0)=1 唯一确定，
    下面用降序排序求出生效分量个数 ρ 后闭式得到 θ。
    """
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]                      # 降序
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0                 # 仍为正的分量
    rho = ind[cond][-1]                       # 生效分量个数
    theta = cssv[cond][-1] / rho              # 阈值
    return np.maximum(v - theta, 0.0)


def objective(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, lam: float) -> float:
    """加权和标量化目标 f(w) = ½ wᵀΣw − λ·μᵀw。"""
    return float(0.5 * w @ Sigma @ w - lam * (mu @ w))


def projected_gradient_descent(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    w0: np.ndarray | None = None,
    max_iter: int = 5000,
    tol: float = 1e-10,
):
    """对给定风险偏好 λ，在 w≥0、Σw=1 约束下最小化 f(w)。

    参数
    ----
    lam      : 风险偏好系数。λ=0 → 长仓最小方差；λ 越大越偏向高收益。
    w0       : 初始点（默认等权）。
    max_iter : 最大迭代步数。
    tol      : 相邻迭代目标值变化阈值（收敛判据）。

    返回
    ----
    (w, history)：最优权重，以及每步目标值列表（用于画收敛曲线）。
    """
    n = len(mu)
    w = np.full(n, 1.0 / n) if w0 is None else project_to_simplex(np.asarray(w0, float))

    # 步长由 Σ 的最大特征值（梯度的利普希茨常数）确定
    L = float(np.linalg.eigvalsh(Sigma).max())
    step = 1.0 / L

    history = [objective(w, mu, Sigma, lam)]
    for _ in range(max_iter):
        grad = Sigma @ w - lam * mu
        w = project_to_simplex(w - step * grad)
        history.append(objective(w, mu, Sigma, lam))
        if abs(history[-1] - history[-2]) < tol:
            break
    return w, history


def numeric_frontier(mu: np.ndarray, Sigma: np.ndarray, lambdas: np.ndarray):
    """扫描一组 λ，得到长仓（w≥0）有效前沿。

    返回 (rets, vols, weights)。利用 warm start（上一解作为下一步初值）加速。
    """
    rets, vols, weights = [], [], []
    w_prev = None
    for lam in lambdas:
        w, _ = projected_gradient_descent(mu, Sigma, lam, w0=w_prev)
        w_prev = w
        rets.append(float(mu @ w))
        vols.append(float(np.sqrt(w @ Sigma @ w)))
        weights.append(w)
    return np.array(rets), np.array(vols), np.array(weights)
