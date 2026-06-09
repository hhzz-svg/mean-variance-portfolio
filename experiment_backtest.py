# -*- coding: utf-8 -*-
"""
实验：均值-方差组合的样本外滚动回测。

把项目的“样本内最优”演示升级为可量化的样本外实验，验证经典命题：
**期望收益 μ 极难估计 → 样本外时“最优”组合跑输朴素策略（等权/GMV/收缩）。**

运行：  python experiment_backtest.py
产物：  figures/backtest_*.png（4 张）、results/backtest_summary.json、
        data/returns_long.csv（更长合成序列，运行后生成）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import generate_data as gen
import data_utils as du
import metrics as mt
import plots as pl
import backtest as bt

RF_ANNUAL = 0.03
L = 252          # 估计窗口：1 年
R = 21           # 再平衡间隔：约 1 个月
N_DAYS_LONG = 2520   # 合成 ~10 年日收益，保证足够多次再平衡


def main():
    data_path = os.path.join(HERE, "data", "returns_long.csv")
    fig_dir = os.path.join(HERE, "figures")
    res_dir = os.path.join(HERE, "results")

    # ---- 1. 数据：更长的合成序列（不动 main.py 用的 returns.csv） ----------
    if not os.path.exists(data_path):
        gen.generate_returns(n_days=N_DAYS_LONG).to_csv(data_path)
    df = du.load_returns(data_path)
    names = list(df.columns)
    R_all = df.to_numpy()
    mu_true = gen.true_mu_annual()
    print(f"[数据] {R_all.shape[0]} 个交易日 × {R_all.shape[1]} 只股票  "
          f"(估计窗 L={L}, 再平衡 R={R})")

    # ---- 2. 样本外滚动回测 -------------------------------------------------
    bt_res = bt.run_backtest(R_all, mu_true, L=L, R=R, rf_annual=RF_ANNUAL)
    daily_oos = bt_res["daily_oos"]
    weights_hist = bt_res["weights_hist"]
    labels = list(daily_oos.keys())
    n_rebal = len(bt_res["rebal_points"])
    print(f"[回测] {n_rebal} 次再平衡，样本外 {bt_res['n_oos_days']} 个交易日")

    # ---- 3. 样本内基准（乐观口径，用于对比“夏普塌缩”） ---------------------
    is_res = bt.in_sample_weights(R_all, mu_true, rf_annual=RF_ANNUAL)

    # ---- 4. 指标汇总 -------------------------------------------------------
    perf = {}
    for s in labels:
        oos = mt.annualized_stats(daily_oos[s], RF_ANNUAL)
        is_sharpe = mt.annualized_stats(is_res[s]["daily"], RF_ANNUAL)["sharpe"]
        perf[s] = {
            "oos_ann_ret": oos["ann_ret"],
            "oos_ann_vol": oos["ann_vol"],
            "oos_sharpe": oos["sharpe"],
            "oos_max_drawdown": mt.max_drawdown(daily_oos[s]),
            "avg_turnover": mt.turnover(weights_hist[s]),
            "in_sample_sharpe": is_sharpe,
        }

    print("\n  策略                样本外夏普   样本内夏普   年化收益   年化波动   最大回撤   换手率")
    for s in labels:
        p = perf[s]
        print(f"  {s:<18}{p['oos_sharpe']:>8.3f}{p['in_sample_sharpe']:>11.3f}"
              f"{p['oos_ann_ret']*100:>10.2f}%{p['oos_ann_vol']*100:>9.2f}%"
              f"{p['oos_max_drawdown']*100:>9.2f}%{p['avg_turnover']*100:>8.2f}%")

    # ---- 5. 数值自检 -------------------------------------------------------
    checks = run_self_checks(bt_res, R_all.shape[0])
    print("\n[自检]")
    for k, v in checks.items():
        print(f"   {'OK ' if v['pass'] else 'FAIL'}  {k}: {v['detail']}")

    # ---- 6. 图像 -----------------------------------------------------------
    dropped = pl.plot_backtest_wealth(
        os.path.join(fig_dir, "backtest_wealth.png"), daily_oos)
    if dropped:
        print(f"[图像] 净值图剔除非经济策略（单日亏损>100%）: {'、'.join(dropped)}")
    pl.plot_sharpe_is_vs_oos(
        os.path.join(fig_dir, "backtest_sharpe_is_vs_oos.png"),
        labels,
        [perf[s]["in_sample_sharpe"] for s in labels],
        [perf[s]["oos_sharpe"] for s in labels])
    pl.plot_turnover_and_condition(
        os.path.join(fig_dir, "backtest_turnover_condition.png"),
        labels, [perf[s]["avg_turnover"] for s in labels],
        bt_res["rebal_points"], bt_res["cond_sample"], bt_res["cond_lw"])
    pl.plot_rolling_weights(
        os.path.join(fig_dir, "backtest_rolling_weights.png"), names,
        bt_res["rebal_points"],
        weights_hist["切点（样本μ,Σ）"], weights_hist["GMV（收缩Σ）"],
        "切点（样本μ,Σ）", "GMV（收缩Σ）")
    print(f"[图像] 4 张回测图已写入 {fig_dir}")

    # ---- 7. 导出结果 -------------------------------------------------------
    summary = {
        "config": {
            "n_days": int(R_all.shape[0]), "n_assets": len(names),
            "estimation_window_L": L, "rebalance_R": R,
            "risk_free_annual": RF_ANNUAL, "n_rebalances": n_rebal,
            "n_oos_days": int(bt_res["n_oos_days"]),
        },
        "performance": {s: {k: round(float(v), 6) for k, v in perf[s].items()}
                        for s in labels},
        "wealth_plot_excluded": dropped,
        "self_checks": checks,
    }
    out = os.path.join(res_dir, "backtest_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[结果] 已写入 {out}")

    all_pass = all(v["pass"] for v in checks.values())
    print("\n==> 全部自检通过" if all_pass else "\n==> 存在未通过的自检")
    return 0 if all_pass else 1


def run_self_checks(bt_res, T_total) -> dict:
    """回测正确性自检：无前视、长度自洽、权重和为1、换手非负、等权正确。"""
    checks = {}
    daily_oos = bt_res["daily_oos"]
    weights_hist = bt_res["weights_hist"]
    L, R = bt_res["L"], bt_res["R"]
    rebal = bt_res["rebal_points"]

    # (1) 无前视 + 持有窗精确平铺 [L, T_total)：再平衡点等差、首点=L、相邻间隔=R，
    #     且样本外天数 == T_total - L（窗口无缝衔接、不重叠 ⇒ 不可能用到未来数据）
    gaps_ok = bool(rebal[0] == L and np.all(np.diff(rebal) == R))
    tiling_ok = all(len(daily_oos[s]) == T_total - L for s in daily_oos)
    checks["无前视/持有窗平铺"] = {
        "pass": bool(gaps_ok and tiling_ok),
        "detail": f"首再平衡={int(rebal[0])}(=L), 间隔恒={R}, "
                  f"样本外天数={len(next(iter(daily_oos.values())))}(=T-L={T_total - L})",
    }

    # (2) 每个策略样本外日收益长度一致（拼接无丢失）
    lens = {s: len(v) for s, v in daily_oos.items()}
    checks["样本外长度自洽"] = {
        "pass": bool(len(set(lens.values())) == 1),
        "detail": f"各策略样本外天数={sorted(set(lens.values()))}",
    }

    # (3) 每个再平衡点、每个策略权重和为 1
    max_dev = 0.0
    for s, W in weights_hist.items():
        max_dev = max(max_dev, float(np.max(np.abs(W.sum(axis=1) - 1.0))))
    checks["权重和为1"] = {
        "pass": bool(max_dev < 1e-8),
        "detail": f"max|Σwᵢ − 1|={max_dev:.2e}",
    }

    # (4) 换手率非负
    tos = {s: mt.turnover(W) for s, W in weights_hist.items()}
    min_to = min(tos.values())
    checks["换手率非负"] = {
        "pass": bool(min_to >= 0.0),
        "detail": f"min 平均换手={min_to:.4f} (≥0)",
    }

    # (5) 等权策略权重确实是 1/n（朴素基准无估计）
    W_eq = weights_hist["等权 1/N"]
    n = W_eq.shape[1]
    eq_err = float(np.max(np.abs(W_eq - 1.0 / n)))
    checks["等权=1/n"] = {
        "pass": bool(eq_err < 1e-12),
        "detail": f"max|wᵢ − 1/n|={eq_err:.2e}",
    }

    return checks


if __name__ == "__main__":
    raise SystemExit(main())
