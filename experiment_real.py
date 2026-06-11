# -*- coding: utf-8 -*-
"""
实验七：真实数据验证 —— 六幕结论在真实市场上还成立吗？

前六幕全在合成因子数据上完成。本实验把同一套 walk-forward 协议与同一组策略，
原封不动地搬到 8 只真实美股行业 ETF（XLK/XLF/XLE/XLV/XLP/XLY/XLI/XLU,
2012–2023，data/returns_real.csv）上，与合成数据并排对比，看结论的"复制率"：

  - 1/N 是否依旧难以战胜？
  - 切点（样本 μ）是否依旧样本外失控？
  - Ledoit-Wolf / NLS 收缩是否依旧改善 GMV？
  - μ 收缩（JS / BL）是否依旧把切点往 1/N 拉？

真实数据没有"真 μ"——用全样本均值作"事后 μ"（用了未来信息，作弊上界），
量化"即使完美预知本段的平均收益，能多赚多少"。

数据来源见 build_real_data.py（Yahoo Finance 公开接口，已固化为 CSV）。
运行：  python experiment_real.py
产物：  figures/real_*.png（3 张）、results/real_summary.json
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
import bootstrap as bs

RF_ANNUAL = 0.03
L, R = 252, 21
BLOCK, B_BOOT, SEED_BOOT = 21, 2000, 2024
BENCH = "等权 1/N"

EXTRA_COVS = {
    "rmt": lambda win: du.mp_clipped_covariance(win)[0],
    "factor": lambda win: du.pca_factor_covariance(win)[0],
    "nls": lambda win: du.nls_covariance(win)[0],
}
EXTRA_MUS = {
    "js": lambda win: du.js_shrunk_mu(win)[0],
    "eq": lambda win: du.equilibrium_mu(win, RF_ANNUAL / 252)[0],
}


def build_strategies() -> dict:
    """跨六幕的代表性策略合集（真实数据上同样可算）。"""
    S = bt.STRATEGIES
    return {
        "等权 1/N": S["等权 1/N"],
        "GMV（样本Σ）": S["GMV（样本Σ）"],
        "GMV（收缩Σ）": S["GMV（收缩Σ）"],
        "GMV（NLS）": bt.make_strategy("gmv", "Sigma_nls"),
        "切点（样本μ,Σ）": S["切点（样本μ,Σ）"],
        "切点（收缩Σ）": S["切点（收缩Σ）"],
        "切点（NLS）": bt.make_strategy("tangency", "Sigma_nls"),
        "切点（JSμ）": bt.make_strategy("tangency", "Sigma_lw", "mu_js"),
        "切点（BLμ）": bt.make_strategy("tangency", "Sigma_lw", "mu_eq"),
        "事后切点（全样本μ）": S["Oracle切点（真μ）"],   # 真实数据：μ_true = 全样本均值
    }


def run_on(R_all, mu_for_oracle):
    """在给定收益矩阵上跑回测 + bootstrap，返回 (labels, perf, boot, res)。"""
    res = bt.run_backtest(R_all, mu_for_oracle, L=L, R=R, rf_annual=RF_ANNUAL,
                          strategies=build_strategies(),
                          extra_covs=EXTRA_COVS, extra_mus=EXTRA_MUS)
    labels = list(res["daily_oos"].keys())
    dm = np.column_stack([res["daily_oos"][s] for s in labels])
    boot = bs.joint_sharpe_bootstrap(dm, labels, RF_ANNUAL, BENCH,
                                     block=BLOCK, B=B_BOOT, seed=SEED_BOOT)
    perf = {}
    for i, s in enumerate(labels):
        d = boot["delta_vs_benchmark"][s]
        perf[s] = {
            "oos_sharpe": float(boot["sharpe_point"][i]),
            "sharpe_ci": [float(boot["sharpe_ci"][i, 0]), float(boot["sharpe_ci"][i, 1])],
            "delta_vs_1N": d["delta_point"], "p_value": d["p_value"],
            "avg_turnover": mt.turnover(res["weights_hist"][s]),
            "oos_max_drawdown": mt.max_drawdown(res["daily_oos"][s]),
        }
    return labels, perf, boot, res, dm


def main():
    fig_dir = os.path.join(HERE, "figures")
    res_dir = os.path.join(HERE, "results")
    real_path = os.path.join(HERE, "data", "returns_real.csv")
    synth_path = os.path.join(HERE, "data", "returns_long.csv")

    if not os.path.exists(real_path):
        raise SystemExit("缺少 data/returns_real.csv，请先运行 python build_real_data.py（需联网）")

    # ---- 真实数据 ---------------------------------------------------------
    Rr = du.load_returns(real_path).to_numpy()
    mu_hindsight = du.annualize_mu(du.estimate_mu(Rr))      # 事后 μ（全样本均值）
    print(f"[真实] {Rr.shape[0]} 个交易日 × {Rr.shape[1]} 只 ETF (2012–2023)")
    labels, perf_real, boot_real, res_real, dm_real = run_on(Rr, mu_hindsight)
    n_rebal = len(res_real["rebal_points"])
    print(f"[真实] {n_rebal} 次再平衡，样本外 {res_real['n_oos_days']} 天")

    # ---- 合成数据（同协议同策略，作对照） ---------------------------------
    if not os.path.exists(synth_path):
        gen.generate_returns(n_days=2520).to_csv(synth_path)
    Rs = du.load_returns(synth_path).to_numpy()
    _, perf_synth, _, _, _ = run_on(Rs, gen.true_mu_annual())

    # ---- 表格：真实 vs 合成 ----------------------------------------------
    print("\n  策略                  真实夏普  Δvs1N(p)      合成夏普   换手率   真实最大回撤")
    for s in labels:
        pr, ps = perf_real[s], perf_synth[s]
        print(f"  {s:<20}{pr['oos_sharpe']:>7.3f}{pr['delta_vs_1N']:>8.3f}"
              f"({pr['p_value']:.2f}){ps['oos_sharpe']:>10.3f}"
              f"{pr['avg_turnover']*100:>9.1f}%{pr['oos_max_drawdown']*100:>9.1f}%")

    # ---- 复制率：方向是否一致（相对 1/N 的符号） --------------------------
    repl = []
    for s in labels:
        if s == BENCH:
            continue
        sr = np.sign(perf_real[s]["delta_vs_1N"])
        ss = np.sign(perf_synth[s]["delta_vs_1N"])
        repl.append(sr == ss)
    repl_rate = float(np.mean(repl))
    print(f"\n[复制率] 相对 1/N 的方向一致策略占比: {repl_rate*100:.0f}% "
          f"({sum(repl)}/{len(repl)})")

    # ---- 自检 -------------------------------------------------------------
    checks = run_self_checks(res_real, boot_real, Rr, mu_hindsight, dm_real, labels)
    print("\n[自检]")
    for k, v in checks.items():
        print(f"   {'OK ' if v['pass'] else 'FAIL'}  {k}: {v['detail']}")

    # ---- 图像 -------------------------------------------------------------
    pl.plot_sharpe_forest(
        os.path.join(fig_dir, "real_sharpe_forest.png"),
        labels, boot_real["sharpe_point"], boot_real["sharpe_ci"], benchmark=BENCH)
    pl.plot_backtest_wealth(
        os.path.join(fig_dir, "real_wealth.png"),
        {s: res_real["daily_oos"][s] for s in labels})
    pl.plot_synth_vs_real(
        os.path.join(fig_dir, "real_vs_synth.png"), labels,
        [perf_synth[s]["oos_sharpe"] for s in labels],
        [perf_real[s]["oos_sharpe"] for s in labels])
    print(f"[图像] 3 张图已写入 {fig_dir}")

    # ---- 导出 -------------------------------------------------------------
    summary = {
        "config": {"source": "Yahoo Finance 8 SPDR sector ETFs 2012-2023",
                   "n_days": int(Rr.shape[0]), "n_assets": int(Rr.shape[1]),
                   "estimation_window_L": L, "rebalance_R": R,
                   "n_rebalances": n_rebal, "n_oos_days": int(res_real["n_oos_days"]),
                   "risk_free_annual": RF_ANNUAL,
                   "bootstrap": {"block": BLOCK, "B": B_BOOT, "seed": SEED_BOOT}},
        "replication_rate_vs_1N_sign": round(repl_rate, 4),
        "real": {s: {k: (round(v, 6) if isinstance(v, float) else
                         [round(x, 6) for x in v] if isinstance(v, list) else v)
                     for k, v in perf_real[s].items()} for s in labels},
        "synthetic": {s: {"oos_sharpe": round(perf_synth[s]["oos_sharpe"], 6),
                          "delta_vs_1N": round(perf_synth[s]["delta_vs_1N"], 6)}
                      for s in labels},
        "self_checks": checks,
    }
    out = os.path.join(res_dir, "real_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[结果] 已写入 {out}")

    all_pass = all(v["pass"] for v in checks.values())
    print("\n==> 全部自检通过" if all_pass else "\n==> 存在未通过的自检")
    return 0 if all_pass else 1


def run_self_checks(res, boot, R_all, mu_hind, dm, labels) -> dict:
    """6 项自检：无前视平铺、权重和、CI 含点估计且口径一致、五估计量 PSD、
    事后 μ 用全样本、bootstrap 可复现。"""
    checks = {}
    daily_oos = res["daily_oos"]
    T_total = R_all.shape[0]

    # (1) 无前视 / 持有窗精确平铺
    rebal = res["rebal_points"]
    gaps_ok = bool(rebal[0] == L and np.all(np.diff(rebal) == R))
    tiling_ok = all(len(daily_oos[s]) == T_total - L for s in daily_oos)
    checks["无前视/平铺"] = {
        "pass": bool(gaps_ok and tiling_ok),
        "detail": f"首点={int(rebal[0])}(=L), 样本外天数={len(next(iter(daily_oos.values())))}(=T−L={T_total-L})",
    }

    # (2) 各策略各再平衡点权重和为 1
    max_dev = max(float(np.max(np.abs(W.sum(axis=1) - 1.0)))
                  for W in res["weights_hist"].values())
    checks["权重和为1"] = {"pass": bool(max_dev < 1e-8), "detail": f"max|Σw−1|={max_dev:.1e}"}

    # (3) CI 含点估计且与 metrics 口径一致
    ci_ok = bool(np.all((boot["sharpe_ci"][:, 0] <= boot["sharpe_point"]) &
                        (boot["sharpe_point"] <= boot["sharpe_ci"][:, 1])))
    ref = np.array([mt.annualized_stats(daily_oos[s], RF_ANNUAL)["sharpe"] for s in labels])
    dev = float(np.max(np.abs(ref - boot["sharpe_point"])))
    checks["CI含点估计且口径一致"] = {"pass": bool(ci_ok and dev < 1e-12),
                                     "detail": f"全包含={ci_ok}, max偏差={dev:.1e}"}

    # (4) 真实数据上五种协方差估计逐窗 PSD
    psd_min = np.inf
    for t in res["rebal_points"]:
        win = R_all[t - L:t]
        for fn in (du.sample_covariance, lambda w: du.ledoit_wolf_shrinkage(w)[0],
                   lambda w: du.mp_clipped_covariance(w)[0],
                   lambda w: du.pca_factor_covariance(w)[0],
                   lambda w: du.nls_covariance(w)[0]):
            psd_min = min(psd_min, float(np.linalg.eigvalsh(fn(win)).min()))
    checks["五估计量PSD"] = {"pass": bool(psd_min > -1e-10),
                            "detail": f"全窗口最小特征值={psd_min:.2e} (>0)"}

    # (5) 事后 μ 确实是全样本年化均值（作弊上界，用了未来信息）
    dev_mu = float(np.max(np.abs(mu_hind - du.annualize_mu(du.estimate_mu(R_all)))))
    checks["事后μ=全样本均值"] = {"pass": bool(dev_mu < 1e-12),
                                 "detail": f"max|μ_hind − 全样本μ|={dev_mu:.1e}"}

    # (6) bootstrap 可复现
    b1 = bs.joint_sharpe_bootstrap(dm, labels, RF_ANNUAL, BENCH, block=BLOCK, B=50, seed=SEED_BOOT)
    b2 = bs.joint_sharpe_bootstrap(dm, labels, RF_ANNUAL, BENCH, block=BLOCK, B=50, seed=SEED_BOOT)
    checks["bootstrap可复现"] = {"pass": bool(np.array_equal(b1["sharpe_samples"], b2["sharpe_samples"])),
                                "detail": "同种子 B=50 两遍逐元素相等"}

    return checks


if __name__ == "__main__":
    raise SystemExit(main())
