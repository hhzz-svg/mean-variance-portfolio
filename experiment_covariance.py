# -*- coding: utf-8 -*-
"""
实验三：协方差估计擂台（随机矩阵理论）+ Block Bootstrap 显著性检验。

实验二证明了"元凶是 μ 的估计误差"。本实验从两个方向交叉验证并补完故事：

1. **把 Σ 修到更好能赢吗？** 在同一 walk-forward 协议下，让四种协方差估计
   （样本 / Ledoit-Wolf 收缩 / MP 特征值裁剪 / PCA 因子重构）分别驱动 GMV
   与切点组合同台对比。预期：GMV 档四者差距很小（q=n/T≈0.03 时样本协方差
   本不病态，RMT 是高维武器）；切点档换什么 Σ 都救不回（μ 噪声主导）。
2. **实验二的排名统计上显著吗？** circular block bootstrap（块长=再平衡周期，
   联合重采保留策略间截面相关）给每个策略的样本外夏普加 95% CI，并对
   "策略 − 等权 1/N"的 ΔSharpe 做双侧检验。

运行：  python experiment_covariance.py
产物：  figures/covariance_*.png（3 张）、results/covariance_summary.json
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
L = 252              # 估计窗口：1 年（与实验二一致）
R = 21               # 再平衡间隔：约 1 个月
N_DAYS_LONG = 2520   # 与实验二共用同一条 10 年合成序列
BLOCK = 21           # bootstrap 块长 = 再平衡周期
B_BOOT = 2000
SEED_BOOT = 2024
BENCH = "等权 1/N"

# 额外协方差估计量：{名称: 日频窗口 → 日频 Σ}（年化由引擎统一做）
EXTRA_COVS = {
    "rmt": lambda win: du.mp_clipped_covariance(win)[0],
    "factor": lambda win: du.pca_factor_covariance(win)[0],
}


def build_strategies() -> dict:
    """10 策略注册表：1/N、GMV×4Σ、切点×4Σ、Oracle。复用实验二的公共策略。"""
    S = bt.STRATEGIES
    return {
        "等权 1/N": S["等权 1/N"],
        "GMV（样本Σ）": S["GMV（样本Σ）"],
        "GMV（收缩Σ）": S["GMV（收缩Σ）"],
        "GMV（RMT裁剪Σ）": bt.make_strategy("gmv", "Sigma_rmt"),
        "GMV（因子Σ）": bt.make_strategy("gmv", "Sigma_factor"),
        "切点（样本μ,Σ）": S["切点（样本μ,Σ）"],
        "切点（收缩Σ）": S["切点（收缩Σ）"],
        "切点（RMT裁剪Σ）": bt.make_strategy("tangency", "Sigma_rmt"),
        "切点（因子Σ）": bt.make_strategy("tangency", "Sigma_factor"),
        "Oracle切点（真μ）": S["Oracle切点（真μ）"],
    }


def main():
    data_path = os.path.join(HERE, "data", "returns_long.csv")
    fig_dir = os.path.join(HERE, "figures")
    res_dir = os.path.join(HERE, "results")

    # ---- 1. 数据（与实验二完全相同的序列） --------------------------------
    if not os.path.exists(data_path):
        gen.generate_returns(n_days=N_DAYS_LONG).to_csv(data_path)
    df = du.load_returns(data_path)
    names = list(df.columns)
    R_all = df.to_numpy()
    mu_true = gen.true_mu_annual()
    n = R_all.shape[1]
    print(f"[数据] {R_all.shape[0]} 个交易日 × {n} 只股票  (L={L}, R={R})")

    # ---- 2. 回测：10 策略 × 4 种协方差 ------------------------------------
    strategies = build_strategies()
    bt_res = bt.run_backtest(R_all, mu_true, L=L, R=R, rf_annual=RF_ANNUAL,
                             strategies=strategies, extra_covs=EXTRA_COVS)
    daily_oos = bt_res["daily_oos"]
    labels = list(daily_oos.keys())
    print(f"[回测] {len(bt_res['rebal_points'])} 次再平衡，"
          f"样本外 {bt_res['n_oos_days']} 个交易日，{len(labels)} 个策略")

    # ---- 3. 谱池化：逐窗特征值 / k / σ²_eff（独立循环，引擎保持无感知） ----
    pooled_eig, k_seq, s2_seq = [], [], []
    trace_err = diag_err_rmt = diag_err_f = 0.0
    psd_min_rmt = psd_min_f = np.inf
    lambda_plus = du.mp_lambda_bounds(L, n)[1]
    for t in bt_res["rebal_points"]:
        win = R_all[t - L:t]
        S = du.sample_covariance(win)
        Sig_rmt, info = du.mp_clipped_covariance(win)
        Sig_f, info_f = du.pca_factor_covariance(win)
        pooled_eig.extend(info["eigvals_raw"].tolist())
        k = info["k_signal"]
        k_seq.append(k)
        sig_sum = info["eigvals_raw"][info["eigvals_raw"] > lambda_plus].sum()
        s2_seq.append((n - sig_sum) / (n - k) if k < n else 1.0)
        # 自检素材
        trace_err = max(trace_err, abs(info["eigvals_clipped"].sum() - n))
        diag_err_rmt = max(diag_err_rmt, np.max(np.abs(np.diag(Sig_rmt) - np.diag(S))))
        diag_err_f = max(diag_err_f, np.max(np.abs(np.diag(Sig_f) - np.diag(S))))
        psd_min_rmt = min(psd_min_rmt, np.linalg.eigvalsh(Sig_rmt).min())
        psd_min_f = min(psd_min_f, np.linalg.eigvalsh(Sig_f).min())
    k_seq = np.asarray(k_seq)
    k_mode = int(np.bincount(k_seq).argmax())
    sigma2_eff = float(np.mean(s2_seq))
    print(f"[谱] λ₊={lambda_plus:.3f}  逐窗信号数 k 众数={k_mode}"
          f"（分布 {dict(zip(*[a.tolist() for a in np.unique(k_seq, return_counts=True)]))}）"
          f"  σ²_eff={sigma2_eff:.3f}")

    # ---- 4. Block bootstrap：夏普 CI + ΔSharpe 检验 -----------------------
    daily_matrix = np.column_stack([daily_oos[s] for s in labels])
    boot = bs.joint_sharpe_bootstrap(daily_matrix, labels, RF_ANNUAL, BENCH,
                                     block=BLOCK, B=B_BOOT, seed=SEED_BOOT)
    print(f"[bootstrap] B={B_BOOT}, 块长={BLOCK}, 种子={SEED_BOOT}")

    # ---- 5. 表格 -----------------------------------------------------------
    perf = {}
    for i, s in enumerate(labels):
        d = boot["delta_vs_benchmark"][s]
        perf[s] = {
            "oos_sharpe": float(boot["sharpe_point"][i]),
            "sharpe_ci": [float(boot["sharpe_ci"][i, 0]), float(boot["sharpe_ci"][i, 1])],
            "delta_vs_1N": d["delta_point"],
            "delta_ci": d["ci"],
            "p_value": d["p_value"],
            "avg_turnover": mt.turnover(bt_res["weights_hist"][s]),
            "oos_max_drawdown": mt.max_drawdown(daily_oos[s]),
        }
    print("\n  策略                  样本外夏普   95% CI            Δ vs 1/N    p值     换手率")
    for s in labels:
        p = perf[s]
        print(f"  {s:<20}{p['oos_sharpe']:>8.3f}   [{p['sharpe_ci'][0]:>6.3f},"
              f"{p['sharpe_ci'][1]:>6.3f}]{p['delta_vs_1N']:>10.3f}"
              f"{p['p_value']:>8.3f}{p['avg_turnover']*100:>9.2f}%")

    # ---- 6. 自检 -----------------------------------------------------------
    checks = run_self_checks(bt_res, boot, R_all, mu_true, labels, k_seq, n,
                             trace_err, diag_err_rmt, diag_err_f,
                             psd_min_rmt, psd_min_f, daily_matrix)
    print("\n[自检]")
    for k_, v in checks.items():
        print(f"   {'OK ' if v['pass'] else 'FAIL'}  {k_}: {v['detail']}")

    # ---- 7. 图像 -----------------------------------------------------------
    pl.plot_sharpe_forest(
        os.path.join(fig_dir, "covariance_sharpe_forest.png"),
        labels, boot["sharpe_point"], boot["sharpe_ci"], benchmark=BENCH)
    cond_dict = {
        "样本协方差 S": bt_res["cond_sample"],
        "Ledoit-Wolf 收缩": bt_res["cond_lw"],
        "RMT 裁剪": bt_res["cond_extra"]["rmt"],
        "PCA 因子": bt_res["cond_extra"]["factor"],
    }
    pl.plot_mp_spectrum_and_condition(
        os.path.join(fig_dir, "covariance_mp_spectrum.png"),
        pooled_eig, n / L, lambda_plus, sigma2_eff, k_mode,
        bt_res["rebal_points"], cond_dict)
    j = labels.index(BENCH)
    delta_samples = {s: boot["sharpe_samples"][:, i] - boot["sharpe_samples"][:, j]
                     for i, s in enumerate(labels) if s != BENCH}
    delta_stats = {s: boot["delta_vs_benchmark"][s] for s in labels if s != BENCH}
    pl.plot_delta_sharpe_bootstrap(
        os.path.join(fig_dir, "covariance_delta_sharpe.png"),
        delta_samples, delta_stats, benchmark=BENCH)
    print(f"[图像] 3 张图已写入 {fig_dir}")

    # ---- 8. 导出 -----------------------------------------------------------
    summary = {
        "config": {
            "n_days": int(R_all.shape[0]), "n_assets": n,
            "estimation_window_L": L, "rebalance_R": R,
            "risk_free_annual": RF_ANNUAL,
            "bootstrap": {"block": BLOCK, "B": B_BOOT, "seed": SEED_BOOT,
                          "benchmark": BENCH},
        },
        "spectrum_stats": {
            "lambda_plus": round(lambda_plus, 6),
            "k_signal_mode": k_mode,
            "k_signal_counts": {int(k_): int(c) for k_, c in
                                zip(*np.unique(k_seq, return_counts=True))},
            "sigma2_eff_mean": round(sigma2_eff, 6),
            "cond_median": {lab: round(float(np.median(v)), 3)
                            for lab, v in cond_dict.items()},
        },
        "performance": {s: {k_: (round(v, 6) if isinstance(v, float) else
                                 [round(x, 6) for x in v] if isinstance(v, list) else v)
                            for k_, v in perf[s].items()} for s in labels},
        "self_checks": checks,
    }
    out = os.path.join(res_dir, "covariance_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[结果] 已写入 {out}")

    all_pass = all(v["pass"] for v in checks.values())
    print("\n==> 全部自检通过" if all_pass else "\n==> 存在未通过的自检")
    return 0 if all_pass else 1


def run_self_checks(bt_res, boot, R_all, mu_true, labels, k_seq, n,
                    trace_err, diag_err_rmt, diag_err_f,
                    psd_min_rmt, psd_min_f, daily_matrix) -> dict:
    """8 项自检：估计量数学性质、bootstrap 正确性、与实验二的回归保护。"""
    checks = {}

    # (1) MP 裁剪保迹（归一前 Σλ' = n）
    checks["MP裁剪保迹"] = {
        "pass": bool(trace_err < 1e-10),
        "detail": f"max|Σλ' − n|={trace_err:.2e}",
    }

    # (2) RMT 估计量：PSD 且对角恢复样本方差
    checks["RMT_PSD且对角恢复"] = {
        "pass": bool(psd_min_rmt > -1e-10 and diag_err_rmt < 1e-12),
        "detail": f"min λ={psd_min_rmt:.2e}, max|diag−样本方差|={diag_err_rmt:.2e}",
    }

    # (3) 因子估计量：PSD、对角恢复、0 ≤ k < n 逐窗成立
    k_ok = bool(np.all((k_seq >= 0) & (k_seq < n)))
    checks["因子Σ_PSD且k合法"] = {
        "pass": bool(psd_min_f > -1e-10 and diag_err_f < 1e-12 and k_ok),
        "detail": f"min λ={psd_min_f:.2e}, k∈[{k_seq.min()},{k_seq.max()}] (<n={n})",
    }

    # (4) CI 包含点估计，且点估计与 metrics.annualized_stats 口径一致
    ci_ok = bool(np.all((boot["sharpe_ci"][:, 0] <= boot["sharpe_point"]) &
                        (boot["sharpe_point"] <= boot["sharpe_ci"][:, 1])))
    ref = np.array([mt.annualized_stats(bt_res["daily_oos"][s], RF_ANNUAL)["sharpe"]
                    for s in labels])
    metric_dev = float(np.max(np.abs(ref - boot["sharpe_point"])))
    checks["CI含点估计且口径一致"] = {
        "pass": bool(ci_ok and metric_dev < 1e-12),
        "detail": f"CI 全包含={ci_ok}, max|夏普−metrics 口径|={metric_dev:.2e}",
    }

    # (5) bootstrap 同种子可复现（B=50 跑两遍逐元素相等）
    b1 = bs.joint_sharpe_bootstrap(daily_matrix, labels, RF_ANNUAL, BENCH,
                                   block=BLOCK, B=50, seed=SEED_BOOT)
    b2 = bs.joint_sharpe_bootstrap(daily_matrix, labels, RF_ANNUAL, BENCH,
                                   block=BLOCK, B=50, seed=SEED_BOOT)
    repro = bool(np.array_equal(b1["sharpe_samples"], b2["sharpe_samples"]))
    checks["bootstrap可复现"] = {
        "pass": repro,
        "detail": f"同种子 B=50 两遍逐元素相等={repro}",
    }

    # (6) 回归保护：公共 6 策略与"默认引擎 + 实验二注册表"的样本外收益逐日一致
    ref_res = bt.run_backtest(R_all, mu_true, L=bt_res["L"], R=bt_res["R"],
                              rf_annual=bt_res["rf_annual"])
    common = [s for s in labels if s in ref_res["daily_oos"]]
    max_diff = max(float(np.max(np.abs(bt_res["daily_oos"][s] -
                                       ref_res["daily_oos"][s]))) for s in common)
    checks["回归保护_公共策略一致"] = {
        "pass": bool(max_diff == 0.0),
        "detail": f"{len(common)} 个公共策略 max|日收益差|={max_diff:.1e}",
    }

    # (7) 基准 vs 自身：Δ ≡ 0 且 p = 1
    d0 = boot["delta_vs_benchmark"][BENCH]
    checks["基准自检"] = {
        "pass": bool(d0["delta_point"] == 0.0 and d0["p_value"] == 1.0),
        "detail": f"Δ={d0['delta_point']}, p={d0['p_value']}",
    }

    # (8) 联合矩阵自洽：列数 = 策略数，行数 = 样本外天数
    shape_ok = bool(daily_matrix.shape == (bt_res["n_oos_days"], len(labels)))
    checks["联合矩阵自洽"] = {
        "pass": shape_ok,
        "detail": f"shape={daily_matrix.shape} (期望 ({bt_res['n_oos_days']}, {len(labels)}))",
    }

    return checks


if __name__ == "__main__":
    raise SystemExit(main())
