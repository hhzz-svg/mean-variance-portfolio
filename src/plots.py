# -*- coding: utf-8 -*-
"""所有图像绘制。输入均为年化口径（收益、波动率），标签用中文。"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # 无显示环境下也能出图
import matplotlib.pyplot as plt

# 中文字体（Windows 自带 Microsoft YaHei / SimHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def _pct(x):
    return np.asarray(x) * 100.0


# --------------------------------------------------------------------------
# 图1：有效前沿 + GMV + 切点组合 + 资本市场线
# --------------------------------------------------------------------------
def plot_efficient_frontier(path, rets, vols, gmv, tan, rf):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(_pct(vols), _pct(rets), "-", color="#1f77b4", lw=2.2,
            label="有效前沿（解析，允许卖空）")

    # 资本市场线：从无风险点 (0, rf) 过切点
    x_cml = np.linspace(0, _pct(tan["vol"]) * 1.6, 50)
    slope = (tan["ret"] - rf) / tan["vol"]          # = 最大夏普比
    ax.plot(x_cml, rf * 100 + slope * x_cml, "--", color="#888",
            lw=1.6, label=f"资本市场线（夏普={slope:.2f}）")

    ax.scatter(_pct(gmv["vol"]), _pct(gmv["ret"]), s=110, marker="*",
               color="#d62728", zorder=5, label="全局最小方差 GMV")
    ax.scatter(_pct(tan["vol"]), _pct(tan["ret"]), s=110, marker="D",
               color="#2ca02c", zorder=5, label="切点组合（最大夏普）")
    ax.scatter(0, rf * 100, s=60, marker="o", color="k",
               zorder=5, label=f"无风险资产 r_f={rf*100:.1f}%")

    ax.set_xlabel("年化波动率 σ (%)")
    ax.set_ylabel(r"年化期望收益 $\mu^{\top}w$ (%)")
    ax.set_title("均值-方差有效前沿（双目标：max 收益 / min 风险）")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图2：前沿 + 单资产 + 蒙特卡洛随机组合云（按夏普着色）
# --------------------------------------------------------------------------
def plot_frontier_with_assets(path, rets, vols, mu, Sigma, asset_names, rf,
                              n_random=4000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(mu)
    W = rng.dirichlet(np.ones(n), size=n_random)        # 长仓随机组合
    pr = W @ mu
    pv = np.sqrt(np.einsum("ij,jk,ik->i", W, Sigma, W))
    sharpe = (pr - rf) / pv

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    sc = ax.scatter(_pct(pv), _pct(pr), c=sharpe, cmap="viridis",
                    s=8, alpha=0.55, label="随机长仓组合")
    fig.colorbar(sc, ax=ax, label="夏普比率")
    ax.plot(_pct(vols), _pct(rets), "-", color="#d62728", lw=2.2,
            label="有效前沿（解析）")

    # 单个资产
    asset_vol = np.sqrt(np.diag(Sigma))
    ax.scatter(_pct(asset_vol), _pct(mu), marker="^", s=70,
               color="k", zorder=5, label="单个股票")
    for name, v, r in zip(asset_names, asset_vol, mu):
        ax.annotate(name, (_pct(v), _pct(r)), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("年化波动率 σ (%)")
    ax.set_ylabel("年化期望收益 (%)")
    ax.set_title("可行域、单资产与有效前沿（分散化效应）")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图3：三种代表性组合的权重对比（分组柱状图）
# --------------------------------------------------------------------------
def plot_weights(path, asset_names, portfolios: dict):
    """portfolios: {标签: 权重向量}。"""
    n = len(asset_names)
    labels = list(portfolios.keys())
    x = np.arange(n)
    width = 0.8 / len(labels)
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for k, lab in enumerate(labels):
        ax.bar(x + (k - (len(labels) - 1) / 2) * width,
               _pct(portfolios[lab]), width, label=lab, color=colors[k % 4])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(asset_names, rotation=30, ha="right")
    ax.set_ylabel("权重 (%)")
    ax.set_title("代表性组合的资产权重（负值=卖空）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图4：PGD 收敛曲线 + 数值(长仓)前沿 vs 解析前沿
# --------------------------------------------------------------------------
def plot_pgd_and_frontiers(path, history, f_star,
                           a_rets, a_vols, n_rets, n_vols):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # (a) 收敛：目标值与最优值之差（半对数）
    gap = np.maximum(np.array(history) - f_star, 1e-16)
    axes[0].semilogy(gap, color="#1f77b4", lw=1.8)
    axes[0].set_xlabel("迭代步 k")
    axes[0].set_ylabel("f(w_k) − f*  （对数轴）")
    axes[0].set_title("投影梯度下降收敛性")
    axes[0].grid(alpha=0.3, which="both")

    # (b) 两条前沿对比
    axes[1].plot(_pct(a_vols), _pct(a_rets), "-", color="#1f77b4",
                 lw=2.2, label="解析前沿（允许卖空）")
    axes[1].plot(_pct(n_vols), _pct(n_rets), "o-", color="#d62728",
                 ms=3, lw=1.6, label="数值前沿（长仓 w≥0）")
    axes[1].set_xlabel("年化波动率 σ (%)")
    axes[1].set_ylabel("年化期望收益 (%)")
    axes[1].set_title("约束对前沿的影响：长仓 vs 允许卖空")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图5：相关系数热力图 + 条件数随样本量的变化（样本 vs 收缩）
# --------------------------------------------------------------------------
def plot_covariance_analysis(path, Sigma_sample, asset_names,
                             windows, cond_s_seq, cond_l_seq, n_assets):
    # 相关矩阵（全样本）
    d = 1.0 / np.sqrt(np.diag(Sigma_sample))
    corr = (Sigma_sample * d).T * d

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    im = axes[0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(asset_names)))
    axes[0].set_yticks(range(len(asset_names)))
    axes[0].set_xticklabels(asset_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(asset_names, fontsize=8)
    axes[0].set_title("相关系数矩阵（块状=行业聚类）")
    for i in range(len(asset_names)):
        for j in range(len(asset_names)):
            axes[0].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                         fontsize=6.5, color="black")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # 条件数随估计窗口长度 T 的变化：样本量越小，样本协方差越病态，
    # 而 Ledoit-Wolf 收缩（统计学习正则化）始终保持良态。
    axes[1].semilogy(windows, cond_s_seq, "o-", color="#d62728",
                     label="样本协方差 S")
    axes[1].semilogy(windows, cond_l_seq, "s-", color="#2ca02c",
                     label="Ledoit-Wolf 收缩 Σ_LW")
    axes[1].axvline(n_assets, color="gray", ls=":", lw=1.2)
    axes[1].text(n_assets * 1.05, axes[1].get_ylim()[1] * 0.3,
                 f"n={n_assets}", fontsize=8, color="gray")
    axes[1].set_xlabel("估计窗口长度 T（交易日）")
    axes[1].set_ylabel("条件数 κ = λ_max/λ_min（对数轴）")
    axes[1].set_title("收缩估计何时重要：样本越少越病态")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ==========================================================================
#  样本外滚动回测专用图（实验：估计误差如何摧毁“样本内最优”）
# ==========================================================================
_BT_COLORS = {
    "等权 1/N": "#7f7f7f",
    "GMV（样本Σ）": "#ff7f0e",
    "GMV（收缩Σ）": "#2ca02c",
    "切点（样本μ,Σ）": "#d62728",
    "切点（收缩Σ）": "#9467bd",
    "长仓最小方差": "#1f77b4",
    "Oracle切点（真μ）": "#17becf",
}


def _color(label, fallback="#333"):
    return _BT_COLORS.get(label, fallback)


# --------------------------------------------------------------------------
# 图A：样本外累计净值曲线（头条结果）
# --------------------------------------------------------------------------
def plot_backtest_wealth(path, daily_oos: dict, trading_days=252):
    """daily_oos: {策略名: 样本外日收益数组}。横轴换算为“年”，纵轴对数。

    无杠杆约束的切点策略样本外日收益常跌破 -100%，累计净值会变负/无意义，
    故净值图只画“净值始终为正”的经济可行策略，并把被剔除者标注出来
    （其失败已由夏普/波动/换手/回撤量化）。返回被剔除的策略名列表。
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    dropped = []
    for label, r in daily_oos.items():
        factor = 1.0 + np.asarray(r)
        if np.any(factor <= 0):                 # 单日亏损>100% → 净值非经济
            dropped.append(label)
            continue
        wealth = np.cumprod(factor)
        years = np.arange(len(wealth)) / trading_days
        lw = 2.4 if label.startswith("Oracle") else 1.8
        ls = "--" if label.startswith("Oracle") else "-"
        ax.plot(years, wealth, ls, color=_color(label), lw=lw, label=label)
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("样本外时间（年）")
    ax.set_ylabel("累计净值（初始=1，对数轴）")
    ax.set_title("样本外滚动回测：累计净值（估计误差的真实代价）")
    if dropped:
        ax.text(0.02, 0.02, "已剔除（净值非经济，单日亏损>100%）:\n" + "、".join(dropped),
                transform=ax.transAxes, fontsize=8, color="#b00",
                va="bottom", ha="left")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(path)
    plt.close(fig)
    return dropped


# --------------------------------------------------------------------------
# 图B：样本内 vs 样本外 夏普对比（“塌缩的差距”）
# --------------------------------------------------------------------------
def plot_sharpe_is_vs_oos(path, labels, is_sharpe, oos_sharpe):
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(x - width / 2, is_sharpe, width, label="样本内（乐观）",
           color="#bbbbbb", edgecolor="k", lw=0.5)
    ax.bar(x + width / 2, oos_sharpe, width, label="样本外（真实）",
           color="#d62728", edgecolor="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("年化夏普比率")
    ax.set_title("样本内 vs 样本外夏普：最优组合的夏普在样本外塌缩")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图C：换手率 + Σ̂ 条件数随时间（样本 vs 收缩）
# --------------------------------------------------------------------------
def plot_turnover_and_condition(path, labels, turnovers,
                                rebal_points, cond_sample, cond_lw,
                                trading_days=252):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    # (a) 各策略平均单边换手率（对数轴：最大值可达数千%，否则小值不可见）
    x = np.arange(len(labels))
    vals = _pct(turnovers)
    bars = axes[0].bar(x, np.maximum(vals, 1e-2),
                       color=[_color(l) for l in labels], edgecolor="k", lw=0.5)
    axes[0].set_yscale("log")
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("平均单边换手率 (%，对数轴)")
    axes[0].set_title("换手率：估计噪声越大，换手越凶")
    axes[0].grid(axis="y", alpha=0.3, which="both")

    # (b) 条件数随再平衡时间
    years = np.asarray(rebal_points) / trading_days
    axes[1].semilogy(years, cond_sample, "-", color="#d62728",
                     lw=1.6, label="样本协方差 S")
    axes[1].semilogy(years, cond_lw, "-", color="#2ca02c",
                     lw=1.6, label="Ledoit-Wolf 收缩 Σ_LW")
    axes[1].set_xlabel("样本外时间（年）")
    axes[1].set_ylabel("条件数 κ = λ_max/λ_min（对数轴）")
    axes[1].set_title("逐期协方差估计的病态程度：收缩始终更良态")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# 图D：滚动权重热图（切点-样本 的剧烈摆动 vs GMV-收缩 的稳定）
# --------------------------------------------------------------------------
def plot_rolling_weights(path, asset_names, rebal_points,
                         weights_unstable, weights_stable,
                         label_unstable, label_stable, trading_days=252):
    years = np.asarray(rebal_points) / trading_days
    extent = [years.min(), years.max(), len(asset_names) - 0.5, -0.5]
    vmax = max(np.abs(weights_unstable).max(), np.abs(weights_stable).max())

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    for ax, W, title in (
        (axes[0], weights_unstable, label_unstable),
        (axes[1], weights_stable, label_stable),
    ):
        im = ax.imshow(W.T, aspect="auto", cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, extent=extent, interpolation="nearest")
        ax.set_yticks(range(len(asset_names)))
        ax.set_yticklabels(asset_names, fontsize=8)
        ax.set_title(f"{title}：逐期权重（红=做多 蓝=做空）", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    axes[1].set_xlabel("样本外时间（年）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
