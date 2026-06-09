# -*- coding: utf-8 -*-
"""组合层面的常用指标。"""
from __future__ import annotations

import numpy as np


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """组合期望收益 μᵀw。"""
    return float(w @ mu)


def portfolio_variance(w: np.ndarray, Sigma: np.ndarray) -> float:
    """组合方差 wᵀΣw。"""
    return float(w @ Sigma @ w)


def portfolio_vol(w: np.ndarray, Sigma: np.ndarray) -> float:
    """组合波动率（标准差）。"""
    return float(np.sqrt(portfolio_variance(w, Sigma)))


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float) -> float:
    """夏普比率 (μᵀw − r_f) / √(wᵀΣw)。"""
    vol = portfolio_vol(w, Sigma)
    return float((w @ mu - rf) / vol) if vol > 0 else 0.0


# --------------------------------------------------------------------------
# 样本外指标：直接作用在“已实现的日收益率序列”上（而非 μ/Σ 假设上）
# --------------------------------------------------------------------------
TRADING_DAYS = 252


def annualized_stats(daily_ret: np.ndarray, rf_annual: float = 0.0) -> dict:
    """从一条已实现日收益率序列计算年化收益、年化波动、年化夏普。

    年化口径与项目其余部分一致：均值 ×252、波动 ×√252。
    夏普用年化超额收益 / 年化波动。
    """
    r = np.asarray(daily_ret, dtype=float)
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float((ann_ret - rf_annual) / ann_vol) if ann_vol > 0 else 0.0
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe}


def max_drawdown(daily_ret: np.ndarray) -> float:
    """最大回撤（负数，越接近 0 越好）。基于简单收益累乘的净值曲线。"""
    r = np.asarray(daily_ret, dtype=float)
    wealth = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def turnover(weights_seq: np.ndarray) -> float:
    """平均单边换手率：每个再平衡点 ½·Σ|w_t − w_{t-1}| 的均值。

    weights_seq 形状 (n_rebal, n_assets)，按再平衡顺序排列。
    采用“目标到目标”的简化口径（忽略持有期内的权重漂移）。
    """
    W = np.asarray(weights_seq, dtype=float)
    if W.shape[0] < 2:
        return 0.0
    step_turnover = 0.5 * np.abs(np.diff(W, axis=0)).sum(axis=1)
    return float(step_turnover.mean())
