# -*- coding: utf-8 -*-
"""
样本外滚动回测引擎（walk-forward / out-of-sample）。

动机
----
现有项目全部是“样本内”结果：切点（最大夏普）组合在样本内一骑绝尘。但这正是
Markowitz 模型最著名的陷阱——**期望收益 μ 极难估计**，样本外时估计误差会摧毁
“最优”组合，使它跑输朴素策略（等权 1/N、GMV、收缩协方差）。本模块用滚动窗口
把“样本内演示”升级为可量化的实验，复刻 DeMiguel-Garlappi-Uppal(2009) 的结论。

协议（严格无前视）
------------------
对每个再平衡点 t（从第 L 天起，每 R 天一次）：
  1. 仅用“前 L 天” returns[t-L : t] 估计 μ̂、Σ̂（样本协方差 & Ledoit-Wolf 收缩）；
  2. 各策略据此构造权重 w（每个再平衡点固定到目标权重，持有到下个再平衡点）；
  3. 用“当期及之后” returns[t : t+R] 的**已实现**收益评估组合：日收益 = r_tᵀw。
估计窗 [t-L, t) 与持有窗 [t, t+R) 不重叠 ⇒ 结构上不可能前视。

矩阵算法衔接
------------
每个再平衡点都复用 analytic.chol_solve（Cholesky 解 SPD 系统）与 numeric 的
投影梯度下降；同时记录 Σ̂ 的条件数（样本 vs 收缩）随时间变化，把原先那张静态
条件数图变成与回测表现直接挂钩的动态证据。
"""
from __future__ import annotations

import numpy as np

import analytic as an
import numeric as nm
import data_utils as du


# --------------------------------------------------------------------------
# 策略注册表：每个策略把“一期估计量 est” 映射为权重向量 w（Σwᵢ=1）
# est 字段：mu_hat / Sigma_sample / Sigma_lw （均为年化）、rf、mu_true、n
# --------------------------------------------------------------------------
def _equal_weight(est) -> np.ndarray:
    n = est["n"]
    return np.full(n, 1.0 / n)


def _gmv_sample(est) -> np.ndarray:
    return an.gmv_portfolio(est["mu_hat"], est["Sigma_sample"])["weights"]


def _gmv_lw(est) -> np.ndarray:
    return an.gmv_portfolio(est["mu_hat"], est["Sigma_lw"])["weights"]


def _tangency_sample(est) -> np.ndarray:
    return an.tangency_portfolio(est["mu_hat"], est["Sigma_sample"], est["rf"])["weights"]


def _tangency_lw(est) -> np.ndarray:
    return an.tangency_portfolio(est["mu_hat"], est["Sigma_lw"], est["rf"])["weights"]


def _longonly_minvar(est) -> np.ndarray:
    w, _ = nm.projected_gradient_descent(est["mu_hat"], est["Sigma_lw"], lam=0.0)
    return w


def _oracle_tangency(est) -> np.ndarray:
    """上界：用因子模型的真实 μ（仍用收缩 Σ）——隔离“μ 估计误差”的贡献。"""
    return an.tangency_portfolio(est["mu_true"], est["Sigma_lw"], est["rf"])["weights"]


# 顺序固定，保证可复现与作图标签一致
STRATEGIES = {
    "等权 1/N": _equal_weight,
    "GMV（样本Σ）": _gmv_sample,
    "GMV（收缩Σ）": _gmv_lw,
    "切点（样本μ,Σ）": _tangency_sample,
    "切点（收缩Σ）": _tangency_lw,
    "长仓最小方差": _longonly_minvar,
    "Oracle切点（真μ）": _oracle_tangency,
}


def estimate_window(win: np.ndarray, rf: float, mu_true: np.ndarray) -> dict:
    """对一个估计窗口（T×n 日收益）做年化 μ̂/Σ̂ 估计，返回策略所需的 est 包。"""
    mu_hat = du.annualize_mu(du.estimate_mu(win))
    Sigma_sample = du.annualize_cov(du.sample_covariance(win))
    Slw_d, delta = du.ledoit_wolf_shrinkage(win)
    Sigma_lw = du.annualize_cov(Slw_d)
    return {
        "mu_hat": mu_hat,
        "Sigma_sample": Sigma_sample,
        "Sigma_lw": Sigma_lw,
        "rf": rf,
        "mu_true": mu_true,
        "n": win.shape[1],
        "delta": delta,
    }


def run_backtest(R_all: np.ndarray, mu_true: np.ndarray,
                 L: int = 252, R: int = 21, rf_annual: float = 0.03,
                 strategies: dict | None = None) -> dict:
    """滚动回测主循环。

    参数
    ----
    R_all    : (T_total, n) 已实现日简单收益率。
    mu_true  : (n,) 真实年化期望收益（oracle 用）。
    L        : 估计窗口长度（交易日）。
    R        : 再平衡间隔（交易日）。
    rf_annual: 年化无风险利率。

    返回
    ----
    dict：每策略的样本外日收益流、权重历史，以及逐期条件数/收缩强度与再平衡索引。
    """
    if strategies is None:
        strategies = STRATEGIES
    R_all = np.asarray(R_all, dtype=float)
    T_total, n = R_all.shape

    rebal_points = list(range(L, T_total, R))
    daily_oos = {s: [] for s in strategies}      # 拼接后的样本外日收益
    weights_hist = {s: [] for s in strategies}   # 每个再平衡点的目标权重
    cond_sample, cond_lw, deltas = [], [], []

    for t in rebal_points:
        win = R_all[t - L:t]                      # 估计窗 [t-L, t)，严格在 t 之前
        hold = R_all[t:t + R]                     # 持有窗 [t, t+R)，已实现收益
        if hold.shape[0] == 0:
            continue
        est = estimate_window(win, rf_annual, mu_true)
        cond_sample.append(du.condition_number(est["Sigma_sample"]))
        cond_lw.append(du.condition_number(est["Sigma_lw"]))
        deltas.append(est["delta"])
        for name, fn in strategies.items():
            w = fn(est)
            weights_hist[name].append(w)
            daily_oos[name].extend((hold @ w).tolist())   # 持有期每日组合收益

    return {
        "daily_oos": {s: np.asarray(v) for s, v in daily_oos.items()},
        "weights_hist": {s: np.asarray(v) for s, v in weights_hist.items()},
        "cond_sample": np.asarray(cond_sample),
        "cond_lw": np.asarray(cond_lw),
        "deltas": np.asarray(deltas),
        "rebal_points": np.asarray(rebal_points[:len(cond_sample)]),
        "L": L, "R": R, "rf_annual": rf_annual,
        "n_oos_days": T_total - L,
    }


def in_sample_weights(R_all: np.ndarray, mu_true: np.ndarray,
                      rf_annual: float = 0.03,
                      strategies: dict | None = None) -> dict:
    """用“整段历史”估计并在“整段历史”上评估（乐观的样本内口径）。

    与样本外结果对比，直观展示估计误差导致的“夏普塌缩”。
    """
    if strategies is None:
        strategies = STRATEGIES
    est = estimate_window(np.asarray(R_all, float), rf_annual, mu_true)
    out = {}
    for name, fn in strategies.items():
        w = fn(est)
        out[name] = {"weights": w, "daily": np.asarray(R_all) @ w}
    return out
