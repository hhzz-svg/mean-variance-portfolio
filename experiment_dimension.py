# -*- coding: utf-8 -*-
"""
实验五：维数相变 —— 给 RMT 翻案。

实验三的尴尬结论是"RMT 在 n=8 时帮倒忙"。但那只讲了故事的一半：RMT 的主场
是**高维**。本实验固定估计窗 T=252，把资产数 n 从 8 扫到 200（q=n/T 从 0.03
逼近 0.8），数据生成过程保持同构——1 市场 + 3 行业因子 + 特质噪声，**真实信号
秩恒为 4，与 n 无关**。预期的相变：

  - q 小：样本协方差不病态，四种估计差距小（实验三的情形）；
  - q → 1：样本协方差条件数爆炸，GMV 权重失控、实现波动飙升；
    而 RMT/因子估计只保留 O(1) 个信号特征对，**维数免疫**。

评价指标用 **GMV 的样本外实现波动**（GMV 的目标函数就是最小化方差，谁的 Σ̂
更接近真相，谁的实现波动更低——不经过 μ，干净隔离 Σ 的质量）。

运行：  python experiment_dimension.py
产物：  figures/dimension_*.png（2 张）、results/dimension_summary.json
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

RF_ANNUAL = 0.03
L, R = 252, 21
N_DAYS = 2520
N_LIST = [8, 16, 25, 50, 100, 150, 200]
SEED_UNIVERSE = 7
SPECTRUM_N = 100          # 用哪个 n 的宇宙画 MP 谱图

EXTRA_COVS = {
    "rmt": lambda win: du.mp_clipped_covariance(win)[0],
    "factor": lambda win: du.pca_factor_covariance(win)[0],
}
GMV_LABELS = {            # Σ估计量显示名 → 策略
    "样本协方差 S": "GMV（样本Σ）",
    "Ledoit-Wolf 收缩": "GMV（收缩Σ）",
    "RMT 裁剪": "GMV（RMT裁剪Σ）",
    "PCA 因子": "GMV（因子Σ）",
}


def build_strategies() -> dict:
    S = bt.STRATEGIES
    return {
        "等权 1/N": S["等权 1/N"],
        "GMV（样本Σ）": S["GMV（样本Σ）"],
        "GMV（收缩Σ）": S["GMV（收缩Σ）"],
        "GMV（RMT裁剪Σ）": bt.make_strategy("gmv", "Sigma_rmt"),
        "GMV（因子Σ）": bt.make_strategy("gmv", "Sigma_factor"),
    }


def run_one_universe(n: int) -> dict:
    """对资产数 n 的合成宇宙跑一遍滚动回测，返回该 n 的全部指标。"""
    df, alpha = gen.generate_universe(n, n_days=N_DAYS, seed=SEED_UNIVERSE)
    R_all = df.to_numpy()
    res = bt.run_backtest(R_all, alpha, L=L, R=R, rf_annual=RF_ANNUAL,
                          strategies=build_strategies(), extra_covs=EXTRA_COVS)
    out = {"n": n, "q": n / L, "res": res, "R_all": R_all}
    out["vol"] = {lab: mt.annualized_stats(res["daily_oos"][s], RF_ANNUAL)["ann_vol"]
                  for lab, s in GMV_LABELS.items()}
    out["sharpe"] = {lab: mt.annualized_stats(res["daily_oos"][s], RF_ANNUAL)["sharpe"]
                     for lab, s in GMV_LABELS.items()}
    out["vol_1n"] = mt.annualized_stats(res["daily_oos"]["等权 1/N"], RF_ANNUAL)["ann_vol"]
    out["cond_median"] = {
        "样本协方差 S": float(np.median(res["cond_sample"])),
        "Ledoit-Wolf 收缩": float(np.median(res["cond_lw"])),
        "RMT 裁剪": float(np.median(res["cond_extra"]["rmt"])),
        "PCA 因子": float(np.median(res["cond_extra"]["factor"])),
    }
    return out


def main():
    fig_dir = os.path.join(HERE, "figures")
    res_dir = os.path.join(HERE, "results")

    # ---- 1. 维数扫描 -------------------------------------------------------
    results = []
    print(f"[扫描] n ∈ {N_LIST}，T={L} 固定，每宇宙 {N_DAYS} 天")
    print("\n  n     q      GMV样本外年化波动:  样本Σ    收缩Σ    RMT     因子    | 等权")
    for n in N_LIST:
        t0 = time.time()
        out = run_one_universe(n)
        results.append(out)
        v = out["vol"]
        print(f"  {n:<5}{out['q']:.3f}   "
              f"{v['样本协方差 S']*100:>16.2f}%{v['Ledoit-Wolf 收缩']*100:>8.2f}%"
              f"{v['RMT 裁剪']*100:>8.2f}%{v['PCA 因子']*100:>8.2f}%"
              f"{out['vol_1n']*100:>9.2f}%   ({time.time()-t0:.1f}s)")

    # ---- 2. n=SPECTRUM_N 的谱池化（画 MP 谱图，与实验三 n=8 版对照） --------
    spec = next(o for o in results if o["n"] == SPECTRUM_N)
    R_spec = spec["R_all"]
    lam_plus = du.mp_lambda_bounds(L, SPECTRUM_N)[1]
    pooled_eig, k_seq, s2_seq = [], [], []
    for t in spec["res"]["rebal_points"]:
        _, info = du.mp_clipped_covariance(R_spec[t - L:t])
        ev = info["eigvals_raw"]
        pooled_eig.extend(ev.tolist())
        k = info["k_signal"]
        k_seq.append(k)
        s2_seq.append((SPECTRUM_N - ev[ev > lam_plus].sum()) / (SPECTRUM_N - k))
    k_seq = np.asarray(k_seq)
    k_mode = int(np.bincount(k_seq).argmax())
    sigma2_eff = float(np.mean(s2_seq))
    print(f"\n[谱@n={SPECTRUM_N}] λ+={lam_plus:.3f}  k 众数={k_mode}  σ²_eff={sigma2_eff:.3f}")

    # ---- 3. 自检 -----------------------------------------------------------
    checks = run_self_checks(results, k_seq)
    print("\n[自检]")
    for k_, v in checks.items():
        print(f"   {'OK ' if v['pass'] else 'FAIL'}  {k_}: {v['detail']}")

    # ---- 4. 图像 -----------------------------------------------------------
    vol_dict = {lab: [o["vol"][lab] for o in results] for lab in GMV_LABELS}
    cond_dict = {lab: [o["cond_median"][lab] for o in results] for lab in GMV_LABELS}
    pl.plot_dimension_phase(
        os.path.join(fig_dir, "dimension_phase.png"),
        N_LIST, vol_dict, cond_dict, T=L,
        vol_1n=[o["vol_1n"] for o in results])
    cond_dict_spec = {
        "样本协方差 S": spec["res"]["cond_sample"],
        "Ledoit-Wolf 收缩": spec["res"]["cond_lw"],
        "RMT 裁剪": spec["res"]["cond_extra"]["rmt"],
        "PCA 因子": spec["res"]["cond_extra"]["factor"],
    }
    pl.plot_mp_spectrum_and_condition(
        os.path.join(fig_dir, "dimension_spectrum.png"),
        pooled_eig, SPECTRUM_N / L, lam_plus, sigma2_eff, k_mode,
        spec["res"]["rebal_points"], cond_dict_spec)
    print(f"[图像] 2 张图已写入 {fig_dir}")

    # ---- 5. 导出 -----------------------------------------------------------
    summary = {
        "config": {"T": L, "rebalance_R": R, "n_days": N_DAYS,
                   "n_list": N_LIST, "seed_universe": SEED_UNIVERSE,
                   "risk_free_annual": RF_ANNUAL},
        "sweep": [{
            "n": o["n"], "q": round(o["q"], 4),
            "gmv_oos_vol": {lab: round(float(v), 6) for lab, v in o["vol"].items()},
            "gmv_oos_sharpe": {lab: round(float(v), 6) for lab, v in o["sharpe"].items()},
            "vol_1n": round(float(o["vol_1n"]), 6),
            "cond_median": {lab: round(v, 2) for lab, v in o["cond_median"].items()},
        } for o in results],
        "spectrum_at_n": {"n": SPECTRUM_N, "lambda_plus": round(lam_plus, 4),
                          "k_signal_mode": k_mode,
                          "sigma2_eff_mean": round(sigma2_eff, 4)},
        "self_checks": checks,
    }
    out_path = os.path.join(res_dir, "dimension_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[结果] 已写入 {out_path}")

    all_pass = all(v["pass"] for v in checks.values())
    print("\n==> 全部自检通过" if all_pass else "\n==> 存在未通过的自检")
    return 0 if all_pass else 1


def run_self_checks(results, k_seq_spec) -> dict:
    """6 项自检：满秩、λ+ 单调、可复现、信号数合理、高维病态排序、权重和。"""
    checks = {}

    # (1) 全部 n 的样本协方差逐窗满秩（条件数有限 ⇔ 最小特征值 > 0）
    max_cond = max(float(np.max(o["res"]["cond_sample"])) for o in results)
    checks["样本Σ满秩"] = {
        "pass": bool(np.isfinite(max_cond)),
        "detail": f"全扫描最大条件数={max_cond:.2e}（有限 ⇔ 正定）",
    }

    # (2) λ+ 随 q 严格单调增
    lps = [du.mp_lambda_bounds(252, o["n"])[1] for o in results]
    checks["λ+单调增"] = {
        "pass": bool(np.all(np.diff(lps) > 0)),
        "detail": f"λ+: {lps[0]:.2f} → {lps[-1]:.2f}",
    }

    # (3) 可复现：n=8 宇宙重新生成+重跑，样本外收益逐日一致
    o8 = next(o for o in results if o["n"] == 8)
    o8_re = run_one_universe(8)
    dev = max(float(np.max(np.abs(o8["res"]["daily_oos"][s] -
                                  o8_re["res"]["daily_oos"][s])))
              for s in o8["res"]["daily_oos"])
    checks["种子可复现"] = {
        "pass": bool(dev == 0.0),
        "detail": f"n=8 重跑 max|日收益差|={dev:.1e}",
    }

    # (4) 信号特征值个数合理（真实公共因子数=4，估计 k 应在 [1, 8] 内）
    checks["信号数合理"] = {
        "pass": bool(k_seq_spec.min() >= 1 and k_seq_spec.max() <= 8),
        "detail": f"n={SPECTRUM_N} 逐窗 k∈[{k_seq_spec.min()},{k_seq_spec.max()}]（真值 4）",
    }

    # (5) 高维病态排序：最大 n 时 cond(样本) > cond(LW) > 不要求，cond(样本) 最大
    big = results[-1]
    cs, cl = big["cond_median"]["样本协方差 S"], big["cond_median"]["Ledoit-Wolf 收缩"]
    checks["高维病态排序"] = {
        "pass": bool(cs > cl),
        "detail": f"n={big['n']}: 中位κ 样本={cs:.0f} > LW={cl:.0f}",
    }

    # (6) 全扫描权重和为 1
    max_dev = 0.0
    for o in results:
        for W in o["res"]["weights_hist"].values():
            max_dev = max(max_dev, float(np.max(np.abs(W.sum(axis=1) - 1.0))))
    checks["权重和为1"] = {
        "pass": bool(max_dev < 1e-8),
        "detail": f"max|Σw − 1|={max_dev:.1e}",
    }

    return checks


if __name__ == "__main__":
    raise SystemExit(main())
