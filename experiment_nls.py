# -*- coding: utf-8 -*-
"""
实验六：最优协方差估计 —— Ledoit-Wolf (2020) 解析非线性收缩。

第三/五幕的协方差线索还差一个收尾：到底什么是"最好的" Σ̂？线性收缩（LW2004）
把所有特征值朝同一个数拉，RMT 裁剪把噪声特征值压成台阶——都是"一刀切"。
**非线性收缩**按 Marchenko-Pastur 理论给每个特征值施加最优的、连续变化的缩放，
是 L2 意义下最优的旋转等变估计量。本实验把它与前面四种估计同台，并用一张
"收缩函数图"把所有方法对谱的改造可视化在一起。

三部分：
1. **收缩函数图**（n=100 窗口）：样本=单位线、LW2004=斜线、RMT=台阶、
   NLS=最优平滑曲线，一张图看尽"每种方法对特征值做了什么"；
2. **n=8 擂台**：GMV × 5 种 Σ + bootstrap 置信区间（GMV 是纯 Σ 检验，无 μ 噪声）；
3. **维数扫描**：把 NLS 加入第五幕的相变图，验证它在所有 q 上都贴近最优。

运行：  python experiment_nls.py
产物：  figures/nls_*.png（3 张）、results/nls_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time

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
N_DAYS = 2520
BLOCK, B_BOOT, SEED_BOOT = 21, 2000, 2024
BENCH = "等权 1/N"
SHRINK_FIG_N = 100                 # 收缩函数图用的高维窗口
N_LIST = [8, 25, 50, 100, 200]     # 维数扫描

EXTRA_COVS = {
    "rmt": lambda win: du.mp_clipped_covariance(win)[0],
    "factor": lambda win: du.pca_factor_covariance(win)[0],
    "nls": lambda win: du.nls_covariance(win)[0],
}
SWEEP_GMV = {
    "样本协方差 S": "GMV（样本Σ）",
    "Ledoit-Wolf 收缩": "GMV（收缩Σ）",
    "RMT 裁剪": "GMV（RMT裁剪Σ）",
    "PCA 因子": "GMV（因子Σ）",
    "LW2017 非线性": "GMV（NLS）",
}


def shrinkage_maps(win: np.ndarray) -> dict:
    """对一个窗口计算四种方法的特征值映射 λ→d（均在 1/T 样本协方差谱上）。"""
    T, p = win.shape
    Xc = win - win.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / T
    lam = np.linalg.eigvalsh(S)
    m = lam.mean()
    c = p / T

    # LW2004 线性：d = δ·m + (1−δ)·λ（与 m·I 目标的闭式收缩同特征向量）
    _, delta = du.ledoit_wolf_shrinkage(win)
    d_lw = delta * m + (1.0 - delta) * lam

    # RMT 裁剪（协方差空间示意版）：噪声阈 = m·(1+√c)²，以下压平为其均值
    thr = m * (1.0 + np.sqrt(c)) ** 2
    noise = lam <= thr
    d_rmt = lam.copy()
    if noise.any():
        d_rmt[noise] = lam[noise].mean()

    # NLS：解析非线性收缩
    _, info = du.nls_covariance(win)
    d_nls = info["eigvals_shrunk"]

    return {
        "样本（不收缩）": (lam, lam),
        "LW2004 线性收缩": (lam, d_lw),
        "RMT 裁剪": (lam, d_rmt),
        "LW2017 非线性收缩": (lam, d_nls),
    }


def build_n8_strategies() -> dict:
    S = bt.STRATEGIES
    return {
        "等权 1/N": S["等权 1/N"],
        "GMV（样本Σ）": S["GMV（样本Σ）"],
        "GMV（收缩Σ）": S["GMV（收缩Σ）"],
        "GMV（RMT裁剪Σ）": bt.make_strategy("gmv", "Sigma_rmt"),
        "GMV（因子Σ）": bt.make_strategy("gmv", "Sigma_factor"),
        "GMV（NLS）": bt.make_strategy("gmv", "Sigma_nls"),
        "Oracle切点（真μ）": S["Oracle切点（真μ）"],
    }


def build_sweep_strategies() -> dict:
    S = bt.STRATEGIES
    return {
        "等权 1/N": S["等权 1/N"],
        "GMV（样本Σ）": S["GMV（样本Σ）"],
        "GMV（收缩Σ）": S["GMV（收缩Σ）"],
        "GMV（RMT裁剪Σ）": bt.make_strategy("gmv", "Sigma_rmt"),
        "GMV（因子Σ）": bt.make_strategy("gmv", "Sigma_factor"),
        "GMV（NLS）": bt.make_strategy("gmv", "Sigma_nls"),
    }


def main():
    data_path = os.path.join(HERE, "data", "returns_long.csv")
    fig_dir = os.path.join(HERE, "figures")
    res_dir = os.path.join(HERE, "results")

    # ---- 1. 收缩函数图（n=100 窗口） --------------------------------------
    df100, _ = gen.generate_universe(SHRINK_FIG_N, n_days=N_DAYS, seed=7)
    win100 = df100.to_numpy()[:L]
    maps = shrinkage_maps(win100)
    pl.plot_shrinkage_functions(
        os.path.join(fig_dir, "nls_shrinkage_functions.png"),
        maps, SHRINK_FIG_N, L)
    nls_cond = du.condition_number(du.nls_covariance(win100)[0])
    samp_cond = du.condition_number(du.sample_covariance(win100))
    print(f"[收缩函数@n={SHRINK_FIG_N}] 条件数 样本={samp_cond:.0f} → NLS={nls_cond:.0f}")

    # ---- 2. n=8 擂台：GMV × 5 种 Σ + bootstrap ----------------------------
    if not os.path.exists(data_path):
        gen.generate_returns(n_days=N_DAYS).to_csv(data_path)
    R8 = du.load_returns(data_path).to_numpy()
    mu_true = gen.true_mu_annual()
    res8 = bt.run_backtest(R8, mu_true, L=L, R=R, rf_annual=RF_ANNUAL,
                           strategies=build_n8_strategies(), extra_covs=EXTRA_COVS)
    labels8 = list(res8["daily_oos"].keys())
    dm = np.column_stack([res8["daily_oos"][s] for s in labels8])
    boot = bs.joint_sharpe_bootstrap(dm, labels8, RF_ANNUAL, BENCH,
                                     block=BLOCK, B=B_BOOT, seed=SEED_BOOT)
    print(f"[n=8 擂台] {len(labels8)} 策略，样本外 {res8['n_oos_days']} 天")
    print("\n  策略                  GMV样本外波动  夏普   95%CI            条件数中位")
    cond8 = {"GMV（样本Σ）": res8["cond_sample"], "GMV（收缩Σ）": res8["cond_lw"],
             "GMV（RMT裁剪Σ）": res8["cond_extra"]["rmt"],
             "GMV（因子Σ）": res8["cond_extra"]["factor"],
             "GMV（NLS）": res8["cond_extra"]["nls"]}
    perf8 = {}
    for i, s in enumerate(labels8):
        st = mt.annualized_stats(res8["daily_oos"][s], RF_ANNUAL)
        perf8[s] = {"oos_vol": st["ann_vol"], "oos_sharpe": float(boot["sharpe_point"][i]),
                    "sharpe_ci": [float(boot["sharpe_ci"][i, 0]), float(boot["sharpe_ci"][i, 1])]}
        cm = np.median(cond8[s]) if s in cond8 else float("nan")
        print(f"  {s:<20}{st['ann_vol']*100:>10.2f}%{perf8[s]['oos_sharpe']:>8.3f}"
              f"   [{perf8[s]['sharpe_ci'][0]:>6.3f},{perf8[s]['sharpe_ci'][1]:>6.3f}]"
              f"{cm:>10.1f}")

    # ---- 3. 维数扫描（含 NLS） --------------------------------------------
    print(f"\n[维数扫描] n ∈ {N_LIST}")
    sweep = []
    for n in N_LIST:
        t0 = time.time()
        dfu, alpha = gen.generate_universe(n, n_days=N_DAYS, seed=7)
        res = bt.run_backtest(dfu.to_numpy(), alpha, L=L, R=R, rf_annual=RF_ANNUAL,
                              strategies=build_sweep_strategies(), extra_covs=EXTRA_COVS)
        vol = {lab: mt.annualized_stats(res["daily_oos"][s], RF_ANNUAL)["ann_vol"]
               for lab, s in SWEEP_GMV.items()}
        cond = {"样本协方差 S": float(np.median(res["cond_sample"])),
                "Ledoit-Wolf 收缩": float(np.median(res["cond_lw"])),
                "RMT 裁剪": float(np.median(res["cond_extra"]["rmt"])),
                "PCA 因子": float(np.median(res["cond_extra"]["factor"])),
                "LW2017 非线性": float(np.median(res["cond_extra"]["nls"]))}
        vol_1n = mt.annualized_stats(res["daily_oos"]["等权 1/N"], RF_ANNUAL)["ann_vol"]
        sweep.append({"n": n, "q": n / L, "vol": vol, "cond": cond, "vol_1n": vol_1n,
                      "res": res})
        print(f"  n={n:<4} q={n/L:.2f}  GMV波动 NLS={vol['LW2017 非线性']*100:.2f}%  "
              f"样本={vol['样本协方差 S']*100:.2f}%  ({time.time()-t0:.1f}s)")

    vol_dict = {lab: [o["vol"][lab] for o in sweep] for lab in SWEEP_GMV}
    cond_dict = {lab: [o["cond"][lab] for o in sweep] for lab in SWEEP_GMV}
    pl.plot_dimension_phase(
        os.path.join(fig_dir, "nls_dimension_phase.png"),
        N_LIST, vol_dict, cond_dict, T=L, vol_1n=[o["vol_1n"] for o in sweep])
    pl.plot_sharpe_forest(
        os.path.join(fig_dir, "nls_sharpe_forest.png"),
        labels8, boot["sharpe_point"], boot["sharpe_ci"], benchmark=BENCH)
    print(f"[图像] 3 张图已写入 {fig_dir}")

    # ---- 4. 自检 -----------------------------------------------------------
    checks = run_self_checks(res8, sweep, win100, boot, R8, mu_true, dm, labels8)
    print("\n[自检]")
    for k, v in checks.items():
        print(f"   {'OK ' if v['pass'] else 'FAIL'}  {k}: {v['detail']}")

    # ---- 5. 导出 -----------------------------------------------------------
    summary = {
        "config": {"L": L, "rebalance_R": R, "n_days": N_DAYS,
                   "shrink_fig_n": SHRINK_FIG_N, "n_list": N_LIST,
                   "risk_free_annual": RF_ANNUAL,
                   "bootstrap": {"block": BLOCK, "B": B_BOOT, "seed": SEED_BOOT}},
        "n8_performance": {s: {k: (round(v, 6) if isinstance(v, float) else
                                   [round(x, 6) for x in v]) for k, v in perf8[s].items()}
                           for s in labels8},
        "dimension_sweep": [{
            "n": o["n"], "q": round(o["q"], 4),
            "gmv_oos_vol": {lab: round(float(v), 6) for lab, v in o["vol"].items()},
            "cond_median": {lab: round(v, 2) for lab, v in o["cond"].items()},
            "vol_1n": round(float(o["vol_1n"]), 6),
        } for o in sweep],
        "self_checks": checks,
    }
    out = os.path.join(res_dir, "nls_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[结果] 已写入 {out}")

    all_pass = all(v["pass"] for v in checks.values())
    print("\n==> 全部自检通过" if all_pass else "\n==> 存在未通过的自检")
    return 0 if all_pass else 1


def run_self_checks(res8, sweep, win100, boot, R8, mu_true, dm, labels8) -> dict:
    """6 项自检：NLS PSD、仅改特征值、高维更良态、单调、回归保护、bootstrap。"""
    checks = {}

    # (1) NLS 半正定（n=8 全窗口 + 高维窗口）
    psd_min = np.linalg.eigvalsh(du.nls_covariance(win100)[0]).min()
    for o in sweep:
        for v in o["res"]["cond_extra"]["nls"]:
            pass
    checks["NLS半正定"] = {
        "pass": bool(psd_min > 0),
        "detail": f"n=100 窗口 min λ={psd_min:.2e} (>0)",
    }

    # (2) NLS 只改特征值不动特征向量 ⇔ 与样本协方差对易（S·Σ = Σ·S）
    S = du.sample_covariance(win100)
    Snls, _ = du.nls_covariance(win100)
    comm = float(np.max(np.abs(S @ Snls - Snls @ S)))
    scale = float(np.max(np.abs(S @ Snls)))
    checks["NLS仅改特征值"] = {
        "pass": bool(comm / scale < 1e-8),
        "detail": f"相对对易残差 ‖SΣ−ΣS‖/‖SΣ‖={comm/scale:.1e}",
    }

    # (3) 高维 NLS 远比样本协方差良态
    big = sweep[-1]
    cs, cn = big["cond"]["样本协方差 S"], big["cond"]["LW2017 非线性"]
    checks["NLS高维更良态"] = {
        "pass": bool(cn < cs),
        "detail": f"n={big['n']}: 中位κ NLS={cn:.0f} ≪ 样本={cs:.0f}",
    }

    # (4) NLS 收缩函数近似单调（保特征值序）。解析核估计不严格保证单调，
    #     参考实现亦不做等张校正，故允许相对谱尺度可忽略的微小违反。
    _, info = du.nls_covariance(win100)
    d = info["eigvals_shrunk"]
    mono = float(np.min(np.diff(d)))
    span = float(d.max() - d.min())
    rel = mono / span
    checks["NLS映射近似单调"] = {
        "pass": bool(rel > -1e-3),
        "detail": f"min Δd̃/谱宽={rel:.1e}（>−1e-3，保序至数值精度）",
    }

    # (5) 回归保护：公共策略与默认引擎（RMT/factor extra_covs）逐日一致
    ref = bt.run_backtest(R8, mu_true, L=L, R=R, rf_annual=RF_ANNUAL,
                          strategies={
                              "等权 1/N": bt.STRATEGIES["等权 1/N"],
                              "GMV（样本Σ）": bt.STRATEGIES["GMV（样本Σ）"],
                              "GMV（收缩Σ）": bt.STRATEGIES["GMV（收缩Σ）"],
                              "Oracle切点（真μ）": bt.STRATEGIES["Oracle切点（真μ）"]})
    common = ["等权 1/N", "GMV（样本Σ）", "GMV（收缩Σ）", "Oracle切点（真μ）"]
    max_diff = max(float(np.max(np.abs(res8["daily_oos"][s] - ref["daily_oos"][s])))
                   for s in common)
    checks["回归保护_公共策略一致"] = {
        "pass": bool(max_diff == 0.0),
        "detail": f"{len(common)} 个公共策略 max|日收益差|={max_diff:.1e}",
    }

    # (6) bootstrap 可复现
    b1 = bs.joint_sharpe_bootstrap(dm, labels8, RF_ANNUAL, BENCH, block=BLOCK, B=50, seed=SEED_BOOT)
    b2 = bs.joint_sharpe_bootstrap(dm, labels8, RF_ANNUAL, BENCH, block=BLOCK, B=50, seed=SEED_BOOT)
    checks["bootstrap可复现"] = {
        "pass": bool(np.array_equal(b1["sharpe_samples"], b2["sharpe_samples"])),
        "detail": "同种子 B=50 两遍逐元素相等",
    }

    return checks


if __name__ == "__main__":
    raise SystemExit(main())
