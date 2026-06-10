# -*- coding: utf-8 -*-
"""
合成多资产日收益率数据生成（因子模型）。

为了让“现实问题实现”有可信的数据，又保证离线、可复现，这里用一个
**三因子结构 + 行业聚类** 的统计模型来模拟某市场 8 只股票的日收益率：

    r_{i,t} = α_i + β_i · f^mkt_t + γ_i · f^sec(i)_t + ε_{i,t}

其中
    f^mkt_t          市场（系统性）因子，所有股票共享 → 制造正相关；
    f^sec_t          行业因子，仅同行业股票共享 → 制造行业内更高相关；
    ε_{i,t}          特质噪声，互相独立 → 提供分散化空间；
    α_i              个股漂移项（决定期望收益的差异）。

这样生成的样本协方差矩阵天然对称正定、且具备真实市场的“块状相关”结构，
非常适合用来演示均值-方差优化与协方差矩阵的矩阵算法。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 8 只股票，分属 3 个行业（科技 / 金融 / 消费）
ASSETS = [
    ("TECH-A", "科技"),
    ("TECH-B", "科技"),
    ("TECH-C", "科技"),
    ("FIN-A", "金融"),
    ("FIN-B", "金融"),
    ("CONS-A", "消费"),
    ("CONS-B", "消费"),
    ("CONS-C", "消费"),
]

TRADING_DAYS = 252  # 一年交易日数，用于年化

# 个股年化超额漂移 α（即“真实期望收益”）：科技高成长高 α，金融/消费偏稳健。
# 因子与噪声在样本内去均值，故总体期望收益恰为该 α，子窗口估计会偏离 → 估计误差。
ALPHA_ANNUAL = np.array([0.18, 0.15, 0.12, 0.08, 0.06, 0.10, 0.07, 0.05])


def true_mu_annual() -> np.ndarray:
    """因子模型设定的真实年化期望收益向量（样本外回测的 oracle 上界用）。"""
    return ALPHA_ANNUAL.copy()


def generate_returns(n_days: int = 756, seed: int = 42) -> pd.DataFrame:
    """生成 n_days×8 的日收益率 DataFrame（默认 3 年 ≈ 756 个交易日）。

    参数
    ----
    n_days : 交易日数量。
    seed   : 随机种子，保证可复现。

    返回
    ----
    pandas.DataFrame，列名为股票代码，每行是某一交易日的全市场收益率。
    """
    rng = np.random.default_rng(seed)
    names = [a[0] for a in ASSETS]
    sectors = [a[1] for a in ASSETS]
    n = len(ASSETS)

    # --- 因子载荷（贴近真实的取值区间）---------------------------------
    alpha_daily = ALPHA_ANNUAL / TRADING_DAYS

    # 市场 β：多数股票在 0.8~1.3 之间
    beta_mkt = np.array([1.30, 1.15, 1.05, 0.95, 0.85, 0.90, 0.80, 0.75])
    # 行业 β：同行业共享一个行业因子
    beta_sec = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.7, 0.6, 0.5])

    # 因子波动（日频标准差）
    sigma_mkt = 0.010          # 市场因子日波动 ~1%
    sigma_sec = {"科技": 0.012, "金融": 0.007, "消费": 0.006}
    # 特质波动：科技更高
    idio_vol = np.array([0.018, 0.016, 0.015, 0.011, 0.010, 0.012, 0.011, 0.010])

    # --- 生成因子时间序列 ---------------------------------------------
    # 关键：把因子与噪声在样本内去均值，使每只股票的样本期望收益恰好等于其 α，
    # 既保留真实的协方差（块状相关）结构，又让 μ 的估计可解释、可复现，
    # 避免有限样本下均值被噪声淹没（“期望收益极难估计”这一现象本身）。
    f_mkt = rng.normal(0.0, sigma_mkt, size=n_days)
    f_mkt -= f_mkt.mean()
    sector_set = sorted(set(sectors))
    f_sec = {}
    for s in sector_set:
        fs = rng.normal(0.0, sigma_sec[s], size=n_days)
        f_sec[s] = fs - fs.mean()

    # --- 合成各资产收益率 ---------------------------------------------
    returns = np.empty((n_days, n))
    for i in range(n):
        eps = rng.normal(0.0, idio_vol[i], size=n_days)
        eps -= eps.mean()
        returns[:, i] = (
            alpha_daily[i]
            + beta_mkt[i] * f_mkt
            + beta_sec[i] * f_sec[sectors[i]]
            + eps
        )

    df = pd.DataFrame(returns, columns=names)
    df.index.name = "day"
    return df


def asset_sectors() -> dict[str, str]:
    """返回 {股票代码: 行业} 映射，供绘图与结果解读使用。"""
    return {name: sec for name, sec in ASSETS}


def generate_universe(n_assets: int, n_days: int = 2520,
                      seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """生成任意资产数的合成市场（维数扫描实验用）。

    数据生成过程与 generate_returns 同构：1 个市场因子 + 3 个行业因子（轮转分配）
    + 独立特质噪声；个股载荷从与原 8 资产相同量级的固定区间内随机抽取（种子固定）。
    关键性质：**真实公共因子数恒为 4，与 n 无关**——n 增大时只有"名字"变多，
    信号秩不变，这正是检验"维数升高时哪种 Σ̂ 估计先崩溃"的干净设定。

    返回 (日收益 DataFrame, 真实年化 α 向量)。
    """
    rng = np.random.default_rng(seed)
    sector_names = ["科技", "金融", "消费"]
    sectors = [sector_names[i % 3] for i in range(n_assets)]
    names = [f"{sec[:1]}{i:03d}" for i, sec in enumerate(sectors)]

    # 载荷随机抽取（区间与原 8 资产手工取值同量级）
    alpha_annual = rng.uniform(0.04, 0.18, size=n_assets)
    beta_mkt = rng.uniform(0.70, 1.35, size=n_assets)
    beta_sec = rng.uniform(0.40, 0.90, size=n_assets)
    idio_vol = rng.uniform(0.009, 0.019, size=n_assets)
    sigma_mkt = 0.010
    sigma_sec = {"科技": 0.012, "金融": 0.007, "消费": 0.006}

    f_mkt = rng.normal(0.0, sigma_mkt, size=n_days)
    f_mkt -= f_mkt.mean()
    f_sec = {}
    for s in sector_names:
        fs = rng.normal(0.0, sigma_sec[s], size=n_days)
        f_sec[s] = fs - fs.mean()

    returns = np.empty((n_days, n_assets))
    alpha_daily = alpha_annual / TRADING_DAYS
    for i in range(n_assets):
        eps = rng.normal(0.0, idio_vol[i], size=n_days)
        eps -= eps.mean()
        returns[:, i] = (alpha_daily[i] + beta_mkt[i] * f_mkt
                         + beta_sec[i] * f_sec[sectors[i]] + eps)

    df = pd.DataFrame(returns, columns=names)
    df.index.name = "day"
    return df, alpha_annual


if __name__ == "__main__":
    import os
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "data", "returns.csv")
    df = generate_returns()
    df.to_csv(out)
    print(f"已生成合成日收益率: {df.shape[0]} 天 × {df.shape[1]} 只股票 -> {out}")
    print("年化平均收益(%):")
    print((df.mean() * TRADING_DAYS * 100).round(2).to_string())
    print("年化波动率(%):")
    print((df.std() * np.sqrt(TRADING_DAYS) * 100).round(2).to_string())
