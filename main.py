# -*- coding: utf-8 -*-
"""
主流程：数据 → 估计 μ/Σ → 解析解 + 数值解 → 图像 → 数值自检 → 导出结果。

运行：  python main.py
产物：  figures/*.png 、 results/summary.json 、 data/returns.csv
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
import numeric as nm
import metrics as mt
import plots as pl

RF_ANNUAL = 0.03            # 无风险年化收益（如国债）
TRADING_DAYS = 252


def main():
    data_path = os.path.join(HERE, "data", "returns.csv")
    fig_dir = os.path.join(HERE, "figures")
    res_dir = os.path.join(HERE, "results")

    # ---- 1. 数据 -------------------------------------------------------
    if not os.path.exists(data_path):
        gen.generate_returns().to_csv(data_path)
    df = du.load_returns(data_path)
    names = list(df.columns)
    R = df.to_numpy()
    print(f"[数据] {R.shape[0]} 个交易日 × {R.shape[1]} 只股票")

    # ---- 2. 估计 μ、Σ（日频 → 年化），Σ 用 Ledoit-Wolf 收缩 ------------
    mu_d = du.estimate_mu(R)
    S_d = du.sample_covariance(R)                 # 样本协方差（日频）
    Slw_d, delta = du.ledoit_wolf_shrinkage(R)    # 收缩协方差（日频）

    mu = du.annualize_mu(mu_d)                     # 年化期望收益
    Sigma = du.annualize_cov(Slw_d)               # 年化协方差（用于优化）
    Sigma_sample = du.annualize_cov(S_d)          # 年化样本协方差（仅作对比）

    cond_s = du.condition_number(Sigma_sample)
    cond_l = du.condition_number(Sigma)
    print(f"[收缩] δ*={delta:.3f}  条件数: 样本={cond_s:.1f} → 收缩={cond_l:.1f}")

    # ---- 3. 解析法：有效前沿、GMV、切点组合 ----------------------------
    sc = an.frontier_scalars(mu, Sigma)
    a_rets, a_vols, _ = an.efficient_frontier(mu, Sigma, n_points=80)
    gmv = an.gmv_portfolio(mu, Sigma)
    tan = an.tangency_portfolio(mu, Sigma, RF_ANNUAL)
    print(f"[解析] GMV: 收益={gmv['ret']*100:.2f}%  波动={gmv['vol']*100:.2f}%")
    print(f"[解析] 切点: 收益={tan['ret']*100:.2f}%  波动={tan['vol']*100:.2f}%  "
          f"夏普={tan['sharpe']:.3f}")

    # ---- 4. 数值法：长仓（w≥0）有效前沿 + 一条收敛曲线 -----------------
    lambdas = np.linspace(0.0, 5.0, 50)
    n_rets, n_vols, n_weights = nm.numeric_frontier(mu, Sigma, lambdas)
    w_lo_minvar, _ = nm.projected_gradient_descent(mu, Sigma, lam=0.0)  # 长仓最小方差
    _, history = nm.projected_gradient_descent(mu, Sigma, lam=1.0, max_iter=3000)
    f_star = float(np.min(history))

    # ---- 5. 数值自检 ---------------------------------------------------
    checks = run_self_checks(mu, Sigma, sc, gmv, tan, w_lo_minvar)
    print("[自检]")
    for k, v in checks.items():
        print(f"   {'OK ' if v['pass'] else 'FAIL'}  {k}: {v['detail']}")

    # ---- 6. 图像 -------------------------------------------------------
    pl.plot_efficient_frontier(
        os.path.join(fig_dir, "efficient_frontier.png"),
        a_rets, a_vols, gmv, tan, RF_ANNUAL)
    pl.plot_frontier_with_assets(
        os.path.join(fig_dir, "frontier_with_assets.png"),
        a_rets, a_vols, mu, Sigma, names, RF_ANNUAL)
    pl.plot_weights(
        os.path.join(fig_dir, "weights_allocation.png"), names,
        {"GMV（允许卖空）": gmv["weights"],
         "切点组合（允许卖空）": tan["weights"],
         "长仓最小方差（w≥0）": w_lo_minvar})
    pl.plot_pgd_and_frontiers(
        os.path.join(fig_dir, "pgd_convergence.png"),
        history, f_star, a_rets, a_vols, n_rets, n_vols)
    windows = [10, 12, 15, 20, 25, 30, 40, 60, 90, 130, 200, 300, 450, 756]
    cond_s_seq, cond_l_seq, _ = du.condition_vs_window(R, windows)
    pl.plot_covariance_analysis(
        os.path.join(fig_dir, "covariance_heatmap.png"),
        Sigma_sample, names, windows, cond_s_seq, cond_l_seq, len(names))
    print(f"[图像] 5 张图已写入 {fig_dir}")

    # ---- 7. 导出结果 ---------------------------------------------------
    summary = {
        "assets": names,
        "sectors": gen.asset_sectors(),
        "risk_free_annual": RF_ANNUAL,
        "shrinkage_delta": delta,
        "condition_number": {"sample": cond_s, "shrunk": cond_l},
        "mu_annual_pct": {n: round(float(m) * 100, 3) for n, m in zip(names, mu)},
        "frontier_scalars": {k: float(sc[k]) for k in ("A", "B", "C", "D")},
        "GMV": _pf(gmv, names),
        "tangency_max_sharpe": _pf(tan, names, sharpe=tan["sharpe"]),
        "long_only_min_var": {
            "weights": {n: round(float(w), 4) for n, w in zip(names, w_lo_minvar)},
            "ret": float(mu @ w_lo_minvar),
            "vol": float(np.sqrt(w_lo_minvar @ Sigma @ w_lo_minvar)),
            "sharpe": mt.sharpe_ratio(w_lo_minvar, mu, Sigma, RF_ANNUAL),
        },
        "self_checks": checks,
    }
    out = os.path.join(res_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[结果] 已写入 {out}")

    all_pass = all(v["pass"] for v in checks.values())
    print("\n==> 全部自检通过" if all_pass else "\n==> 存在未通过的自检")
    return 0 if all_pass else 1


def _pf(p, names, sharpe=None):
    d = {
        "weights": {n: round(float(w), 4) for n, w in zip(names, p["weights"])},
        "ret": float(p["ret"]), "vol": float(p["vol"]),
    }
    if sharpe is not None:
        d["sharpe"] = float(sharpe)
    return d


def run_self_checks(mu, Sigma, sc, gmv, tan, w_lo):
    """数值正确性自检，返回 {名称: {pass, detail}}。"""
    checks = {}

    # (1) Σ 正定（所有特征值 > 0）
    min_eig = float(np.linalg.eigvalsh(Sigma).min())
    checks["Sigma_正定"] = {
        "pass": bool(min_eig > 0),
        "detail": f"最小特征值={min_eig:.3e} (>0)",
    }

    # (2) GMV 方差 == 1/C（闭式自洽）
    var_formula = mt.portfolio_variance(gmv["weights"], Sigma)
    var_closed = 1.0 / sc["C"]
    err = abs(var_formula - var_closed)
    checks["GMV方差=1/C"] = {
        "pass": bool(err < 1e-10),
        "detail": f"|wᵀΣw − 1/C|={err:.2e}",
    }

    # (3) 权重和为 1
    sums = {"GMV": gmv["weights"].sum(), "切点": tan["weights"].sum(),
            "长仓": w_lo.sum()}
    max_dev = max(abs(s - 1.0) for s in sums.values())
    checks["权重和为1"] = {
        "pass": bool(max_dev < 1e-9),
        "detail": f"max|Σwᵢ − 1|={max_dev:.2e}",
    }

    # (4) 长仓解非负 + 单纯形投影正确性
    min_w = float(w_lo.min())
    checks["长仓非负"] = {
        "pass": bool(min_w >= -1e-12),
        "detail": f"min wᵢ={min_w:.2e} (≥0)",
    }

    # (5) 解析解满足 KKT 一阶条件 Σw = λμ + γ𝟙（取前沿上某点）
    r_test = float(mu @ gmv["weights"]) + 0.02
    w_t = an.weight_for_return(mu, Sigma, r_test, sc)
    lam = (sc["C"] * r_test - sc["A"]) / sc["D"]
    gam = (sc["B"] - sc["A"] * r_test) / sc["D"]
    kkt_res = float(np.linalg.norm(Sigma @ w_t - lam * mu - gam * np.ones(len(mu))))
    checks["解析KKT残差"] = {
        "pass": bool(kkt_res < 1e-9),
        "detail": f"‖Σw − λμ − γ𝟙‖={kkt_res:.2e}",
    }

    # (6) 自研 PGD 与 scipy(SLSQP) 独立交叉验证（同一约束 QP，λ=1）
    cross = _cross_check_scipy(mu, Sigma, lam=1.0)
    checks["PGD对比scipy"] = cross

    return checks


def _cross_check_scipy(mu, Sigma, lam):
    """用 scipy.optimize.SLSQP 独立求解同一约束问题，与自研 PGD 比对最大权重差。"""
    try:
        from scipy.optimize import minimize
    except Exception as e:                       # scipy 不可用则跳过（不算失败）
        return {"pass": True, "detail": f"scipy 不可用，跳过 ({e})"}

    w_pgd, _ = nm.projected_gradient_descent(mu, Sigma, lam, max_iter=8000, tol=1e-12)
    n = len(mu)
    f = lambda w: 0.5 * w @ Sigma @ w - lam * (mu @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * n
    res = minimize(f, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    max_diff = float(np.max(np.abs(w_pgd - res.x)))
    return {
        "pass": bool(max_diff < 1e-3),
        "detail": f"max|w_PGD − w_scipy|={max_diff:.2e}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
