# -*- coding: utf-8 -*-
"""
实验四：给 μ 开药方 —— Bayes-Stein 收缩与 Black-Litterman 均衡先验。

前三个实验把诊断做完了：样本内最优的切点组合，样本外被 μ 的估计误差摧毁，
修 Σ 救不了它。本实验检验经典药方——**对 μ 本身做收缩**：

1. **Jorion (1986) Bayes-Stein**：把样本 μ̂ 向 GMV 隐含收益 μ₀·𝟙 收缩，
   强度由马氏距离数据驱动——与 Ledoit-Wolf 对 Σ 的收缩完全对仗；
2. **Black-Litterman 均衡先验**：从"市场组合=等权"反推均衡隐含收益
   π = r_f + δΣw_eq。定理：以 π 为输入的切点组合**恰好还原等权**
   （无观点的 BL = 持有市场），本实验将其作为结构自检数值验证；
3. **收缩强度 φ 扫描**：μ_φ = (1−φ)π + φμ̂ 从纯均衡滑到纯样本，
   看样本外夏普在哪个 φ 达到峰值——"该信数据几分"的直接回答。

协议与实验二/三完全一致（walk-forward，L=252，R=21，严格无前视），
所有切点策略固定用 Ledoit-Wolf Σ，以隔离 μ 的影响。

运行：  python experiment_mu_shrinkage.py
产物：  figures/mu_*.png（3 张）、results/mu_summary.json
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
import analytic as an
import metrics as mt
import plots as pl
import backtest as bt
import bootstrap as bs

RF_ANNUAL = 0.03
L, R = 252, 21
N_DAYS_LONG = 2520
BLOCK, B_BOOT, SEED_BOOT = 21, 2000, 2024
BENCH = "等权 1/N"
PHIS = np.round(np.linspace(0.0, 1.0, 11), 1)

EXTRA_MUS = {
    "js": lambda win: du.js_shrunk_mu(win)[0],
    "eq": lambda win: du.equilibrium_mu(win, RF_ANNUAL / 252)[0],
}

# 主策略（进表格、bootstrap、净值图）
MAIN = ["等权 1/N", "GMV（收缩Σ）", "切点（样本μ）", "切点（James-Steinμ）",
        "切点（均衡μ·BL先验）", "Oracle切点（真μ）"]


def _phi_strategy(phi: float):
    """μ_φ = (1−φ)·π + φ·μ̂ 的切点策略（Σ 固定 LW）。"""
    def _fn(est):
        mu_phi = (1.0 - phi) * est["mu_eq"] + phi * est["mu_hat"]
        return an.tangency_portfolio(mu_phi, est["Sigma_lw"], est["rf"])["weights"]
    return _fn


def build_strategies() -> dict:
    S = bt.STRATEGIES
    strats = {
        "等权 1/N": S["等权 1/N"],
        "GMV（收缩Σ）": S["GMV（收缩Σ）"],
        "切点（样本μ）": bt.make_strategy("tangency", "Sigma_lw"),
        "切点（James-Steinμ）": bt.make_strategy("tangency", "Sigma_lw", "mu_js"),
        "切点（均衡μ·BL先验）": bt.make_strategy("tangency", "Sigma_lw", "mu_eq"),
        "Oracle切点（真μ）": S["Oracle切点（真μ）"],
    }
    for p in PHIS:
        strats[f"φ={p:.1f}"] = _phi_strategy(float(p))
    return strats


def main():
    data_path = os.path.join(HERE, "data", "returns_long.csv")
    fig_dir = os.path.join(HERE, "figures")
    res_dir = os.path.join(HERE, "results")

    # ---- 1. 数据（与实验二/三同一条序列） ----------------------------------
    if not os.path.exists(data_path):
        gen.generate_returns(n_days=N_DAYS_LONG).to_csv(data_path)
    df = du.load_returns(data_path)
    R_all = df.to_numpy()
    mu_true = gen.true_mu_annual()
    print(f"[数据] {R_all.shape[0]} 个交易日 × {R_all.shape[1]} 只股票  (L={L}, R={R})")

    # ---- 2. 回测：6 主策略 + 11 个 φ 扫描点 -------------------------------
    bt_res = bt.run_backtest(R_all, mu_true, L=L, R=R, rf_annual=RF_ANNUAL,
                             strategies=build_strategies(), extra_mus=EXTRA_MUS)
    daily_oos = bt_res["daily_oos"]
    print(f"[回测] {len(bt_res['rebal_points'])} 次再平衡，"
          f"样本外 {bt_res['n_oos_days']} 天，{len(daily_oos)} 个策略（含 φ 扫描）")

    # ---- 3. 逐窗 μ 诊断：估计误差与收缩强度 -------------------------------
    mu_true_d = mu_true / 252.0
    err_hat, err_js, err_eq, w_shrinks = [], [], [], []
    bl_dev = 0.0                                  # BL先验切点 vs 1/N 的最大偏差
    for t in bt_res["rebal_points"]:
        win = R_all[t - L:t]
        mu_js, info = du.js_shrunk_mu(win)
        pi, _ = du.equilibrium_mu(win, RF_ANNUAL / 252)
        err_hat.append(np.linalg.norm(win.mean(axis=0) - mu_true_d) * 252)
        err_js.append(np.linalg.norm(mu_js - mu_true_d) * 252)
        err_eq.append(np.linalg.norm(pi - mu_true_d) * 252)
        w_shrinks.append(info["w_shrink"])
    W_bl = bt_res["weights_hist"]["切点（均衡μ·BL先验）"]
    bl_dev = float(np.max(np.abs(W_bl - 1.0 / R_all.shape[1])))
    w_shrinks = np.asarray(w_shrinks)
    print(f"[诊断] 平均误差(年化): 样本μ̂={np.mean(err_hat):.3f}  "
          f"JS={np.mean(err_js):.3f}  均衡π={np.mean(err_eq):.3f}  "
          f"w_shrink∈[{w_shrinks.min():.2f},{w_shrinks.max():.2f}]")

    # ---- 4. bootstrap（仅主策略） + φ 扫描夏普 ----------------------------
    daily_matrix = np.column_stack([daily_oos[s] for s in MAIN])
    boot = bs.joint_sharpe_bootstrap(daily_matrix, MAIN, RF_ANNUAL, BENCH,
                                     block=BLOCK, B=B_BOOT, seed=SEED_BOOT)
    phi_sharpes = [mt.annualized_stats(daily_oos[f"φ={p:.1f}"], RF_ANNUAL)["sharpe"]
                   for p in PHIS]

    # ---- 5. 表格 -----------------------------------------------------------
    perf = {}
    for i, s in enumerate(MAIN):
        d = boot["delta_vs_benchmark"][s]
        perf[s] = {
            "oos_sharpe": float(boot["sharpe_point"][i]),
            "sharpe_ci": [float(boot["sharpe_ci"][i, 0]), float(boot["sharpe_ci"][i, 1])],
            "delta_vs_1N": d["delta_point"], "delta_ci": d["ci"], "p_value": d["p_value"],
            "avg_turnover": mt.turnover(bt_res["weights_hist"][s]),
            "oos_max_drawdown": mt.max_drawdown(daily_oos[s]),
        }
    print("\n  策略                    样本外夏普   95% CI            Δ vs 1/N    p值     换手率")
    for s in MAIN:
        p = perf[s]
        print(f"  {s:<22}{p['oos_sharpe']:>8.3f}   [{p['sharpe_ci'][0]:>6.3f},"
              f"{p['sharpe_ci'][1]:>6.3f}]{p['delta_vs_1N']:>10.3f}"
              f"{p['p_value']:>8.3f}{p['avg_turnover']*100:>9.2f}%")
    i_best = int(np.argmax(phi_sharpes))
    print(f"\n  φ 扫描: 最优 φ*={PHIS[i_best]:.1f}  夏普={phi_sharpes[i_best]:.3f}  "
          f"(φ=0: {phi_sharpes[0]:.3f} → φ=1: {phi_sharpes[-1]:.3f})")

    # ---- 6. 自检 -----------------------------------------------------------
    checks = run_self_checks(bt_res, boot, R_all, mu_true, daily_matrix,
                             w_shrinks, bl_dev, err_hat, err_js)
    print("\n[自检]")
    for k, v in checks.items():
        print(f"   {'OK ' if v['pass'] else 'FAIL'}  {k}: {v['detail']}")

    # ---- 7. 图像 -----------------------------------------------------------
    pl.plot_sharpe_forest(
        os.path.join(fig_dir, "mu_sharpe_forest.png"),
        MAIN, boot["sharpe_point"], boot["sharpe_ci"], benchmark=BENCH)
    pl.plot_mu_diagnosis(
        os.path.join(fig_dir, "mu_diagnosis.png"),
        {"样本μ": err_hat, "James-Stein": err_js, "均衡π": err_eq},
        PHIS, phi_sharpes,
        sharpe_1n=perf[BENCH]["oos_sharpe"],
        sharpe_oracle=perf["Oracle切点（真μ）"]["oos_sharpe"])
    pl.plot_backtest_wealth(
        os.path.join(fig_dir, "mu_wealth.png"),
        {s: daily_oos[s] for s in MAIN})
    print(f"[图像] 3 张图已写入 {fig_dir}")

    # ---- 8. 导出 -----------------------------------------------------------
    summary = {
        "config": {"n_days": int(R_all.shape[0]), "n_assets": int(R_all.shape[1]),
                   "estimation_window_L": L, "rebalance_R": R,
                   "risk_free_annual": RF_ANNUAL,
                   "bootstrap": {"block": BLOCK, "B": B_BOOT, "seed": SEED_BOOT,
                                 "benchmark": BENCH}},
        "mu_diagnosis": {
            "mean_error_annual": {"sample": round(float(np.mean(err_hat)), 4),
                                  "james_stein": round(float(np.mean(err_js)), 4),
                                  "equilibrium": round(float(np.mean(err_eq)), 4)},
            "w_shrink_range": [round(float(w_shrinks.min()), 4),
                               round(float(w_shrinks.max()), 4)],
        },
        "phi_sweep": {f"{p:.1f}": round(float(s), 4)
                      for p, s in zip(PHIS, phi_sharpes)},
        "performance": {s: {k: (round(v, 6) if isinstance(v, float) else
                                [round(x, 6) for x in v] if isinstance(v, list) else v)
                            for k, v in perf[s].items()} for s in MAIN},
        "self_checks": checks,
    }
    out = os.path.join(res_dir, "mu_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[结果] 已写入 {out}")

    all_pass = all(v["pass"] for v in checks.values())
    print("\n==> 全部自检通过" if all_pass else "\n==> 存在未通过的自检")
    return 0 if all_pass else 1


def run_self_checks(bt_res, boot, R_all, mu_true, daily_matrix,
                    w_shrinks, bl_dev, err_hat, err_js) -> dict:
    """8 项自检：收缩强度合法、混合端点一致、BL 定理、回归保护、bootstrap 正确性。"""
    checks = {}
    daily_oos = bt_res["daily_oos"]

    # (1) Jorion 收缩强度 ŵ ∈ (0, 1] 逐窗成立
    checks["JS收缩强度合法"] = {
        "pass": bool(np.all((w_shrinks > 0) & (w_shrinks <= 1))),
        "detail": f"w_shrink∈[{w_shrinks.min():.3f},{w_shrinks.max():.3f}] ⊂ (0,1]",
    }

    # (2) φ 混合端点一致：φ=1 ≡ 切点（样本μ），φ=0 ≡ 切点（均衡μ·BL先验）
    d1 = float(np.max(np.abs(daily_oos["φ=1.0"] - daily_oos["切点（样本μ）"])))
    d0 = float(np.max(np.abs(daily_oos["φ=0.0"] - daily_oos["切点（均衡μ·BL先验）"])))
    checks["φ混合端点一致"] = {
        "pass": bool(d1 < 1e-12 and d0 < 1e-12),
        "detail": f"max|φ=1 − 样本μ̂|={d1:.1e}, max|φ=0 − BL先验|={d0:.1e}",
    }

    # (3) BL 定理：无观点的均衡切点 ≡ 等权 1/N（逐窗权重）
    checks["BL无观点=市场组合"] = {
        "pass": bool(bl_dev < 1e-10),
        "detail": f"max|w_BL − 1/n|={bl_dev:.1e}",
    }

    # (4) 回归保护：与默认引擎的公共策略逐日一致（含改名映射 样本μ̂↔收缩Σ）
    ref = bt.run_backtest(R_all, mu_true, L=bt_res["L"], R=bt_res["R"],
                          rf_annual=bt_res["rf_annual"])
    pairs = {"等权 1/N": "等权 1/N", "GMV（收缩Σ）": "GMV（收缩Σ）",
             "切点（样本μ）": "切点（收缩Σ）", "Oracle切点（真μ）": "Oracle切点（真μ）"}
    max_diff = max(float(np.max(np.abs(daily_oos[a] - ref["daily_oos"][b])))
                   for a, b in pairs.items())
    checks["回归保护_公共策略一致"] = {
        "pass": bool(max_diff == 0.0),
        "detail": f"{len(pairs)} 对公共策略 max|日收益差|={max_diff:.1e}",
    }

    # (5) CI 包含点估计，点估计与 metrics 口径一致
    ci_ok = bool(np.all((boot["sharpe_ci"][:, 0] <= boot["sharpe_point"]) &
                        (boot["sharpe_point"] <= boot["sharpe_ci"][:, 1])))
    import metrics as mt
    ref_sharpe = np.array([mt.annualized_stats(daily_oos[s], 0.03)["sharpe"]
                           for s in boot["labels"]])
    dev = float(np.max(np.abs(ref_sharpe - boot["sharpe_point"])))
    checks["CI含点估计且口径一致"] = {
        "pass": bool(ci_ok and dev < 1e-12),
        "detail": f"CI 全包含={ci_ok}, max偏差={dev:.1e}",
    }

    # (6) bootstrap 同种子可复现
    b1 = bs.joint_sharpe_bootstrap(daily_matrix, boot["labels"], 0.03, BENCH,
                                   block=BLOCK, B=50, seed=SEED_BOOT)
    b2 = bs.joint_sharpe_bootstrap(daily_matrix, boot["labels"], 0.03, BENCH,
                                   block=BLOCK, B=50, seed=SEED_BOOT)
    checks["bootstrap可复现"] = {
        "pass": bool(np.array_equal(b1["sharpe_samples"], b2["sharpe_samples"])),
        "detail": "同种子 B=50 两遍逐元素相等",
    }

    # (7) 基准自检：Δ ≡ 0、p = 1
    d0_ = boot["delta_vs_benchmark"][BENCH]
    checks["基准自检"] = {
        "pass": bool(d0_["delta_point"] == 0.0 and d0_["p_value"] == 1.0),
        "detail": f"Δ={d0_['delta_point']}, p={d0_['p_value']}",
    }

    # (8) JS 确实降低 μ 估计误差（逐窗均值严格更小）
    imp = float(np.mean(err_hat) - np.mean(err_js))
    checks["JS降低μ误差"] = {
        "pass": bool(imp > 0),
        "detail": f"平均误差改善 {imp:.4f}（样本 {np.mean(err_hat):.3f} → JS {np.mean(err_js):.3f}）",
    }

    return checks


if __name__ == "__main__":
    raise SystemExit(main())
