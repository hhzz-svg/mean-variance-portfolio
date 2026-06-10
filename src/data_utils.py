# -*- coding: utf-8 -*-
"""
参数估计：从历史收益率估计期望收益向量 μ 与协方差矩阵 Σ。

包含两种协方差估计：
  1. 样本协方差 S —— 无偏但在维数较高 / 样本较少时病态（条件数大、可能近奇异）。
  2. Ledoit-Wolf 收缩估计 —— 统计学习（machine learning）方法，把样本协方差
     向结构化目标 F = m·I 收缩，自动选取最优收缩强度 δ*，得到良态（well-conditioned）
     的估计。这正是本项目“机器学习相关”的体现：用收缩/正则化改善估计量。

参考：Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional
covariance matrices", Journal of Multivariate Analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_returns(path: str) -> pd.DataFrame:
    """读取收益率 CSV（第一列 day 为索引）。"""
    return pd.read_csv(path, index_col="day")


def estimate_mu(returns: np.ndarray) -> np.ndarray:
    """样本期望收益（按列求均值）。返回形状 (n,)。"""
    return np.asarray(returns).mean(axis=0)


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """样本协方差矩阵 S（无偏估计，分母 T-1）。返回形状 (n, n)。"""
    return np.cov(np.asarray(returns), rowvar=False, bias=False)


def ledoit_wolf_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf 收缩协方差估计，目标为按比例的单位阵 F = m·I。

    估计量： Σ_LW = δ·(m·I) + (1-δ)·S
    其中 m = tr(S)/n 是平均方差，δ ∈ [0,1] 是数据驱动的最优收缩强度。

    采用 Ledoit-Wolf(2004) 的内积 <A,B> = tr(A Bᵀ)/n 推导的闭式 δ。

    返回
    ----
    (Sigma_lw, delta)
    """
    X = np.asarray(returns, dtype=float)
    T, n = X.shape
    # 去均值（收缩公式基于去均值后的二阶矩）
    Xc = X - X.mean(axis=0, keepdims=True)
    # 用 1/T 的样本二阶矩（与 LW 推导一致）
    S = (Xc.T @ Xc) / T

    m = np.trace(S) / n                       # 收缩目标的标量：平均方差
    # d² = (1/n)·||S - m·I||_F²
    d2 = np.sum((S - m * np.eye(n)) ** 2) / n

    # b̄² = (1/T²)·Σ_t (1/n)·||x_t x_tᵀ - S||_F²
    #     用恒等式 ||x x' - S||_F² = ||x||⁴ - 2·xᵀS x + ||S||_F²  做向量化
    norm_x4 = (np.sum(Xc ** 2, axis=1)) ** 2          # 每个样本 ||x_t||⁴
    quad = np.einsum("ti,ij,tj->t", Xc, S, Xc)        # 每个样本 x_tᵀ S x_t
    S_fro2 = np.sum(S ** 2)
    per_t = norm_x4 - 2.0 * quad + S_fro2
    b2_bar = np.sum(per_t) / (n * T * T)

    b2 = min(b2_bar, d2)                      # 截断保证 δ ∈ [0,1]
    delta = 0.0 if d2 == 0 else b2 / d2

    Sigma_lw = delta * m * np.eye(n) + (1.0 - delta) * S
    return Sigma_lw, float(delta)


def annualize_mu(mu_daily: np.ndarray) -> np.ndarray:
    """日频期望收益年化（× 交易日数）。"""
    return np.asarray(mu_daily) * TRADING_DAYS


def annualize_cov(cov_daily: np.ndarray) -> np.ndarray:
    """日频协方差年化（× 交易日数）。"""
    return np.asarray(cov_daily) * TRADING_DAYS


def condition_number(matrix: np.ndarray) -> float:
    """对称正定矩阵的条件数 = λ_max / λ_min（用于对比样本估计与收缩估计的良态性）。"""
    eig = np.linalg.eigvalsh(matrix)
    lo = eig.min()
    return float(eig.max() / lo) if lo > 0 else float("inf")


def condition_vs_window(returns: np.ndarray, windows):
    """计算条件数随估计窗口长度 T 的变化（样本协方差 vs Ledoit-Wolf 收缩）。

    用于展示“高维/短样本时收缩估计才真正重要”这一统计学习结论。
    返回 (cond_sample_list, cond_lw_list, delta_list)。
    """
    cond_s, cond_l, deltas = [], [], []
    for T in windows:
        Rw = np.asarray(returns)[:T]
        S = sample_covariance(Rw)
        Slw, d = ledoit_wolf_shrinkage(Rw)
        cond_s.append(condition_number(S))
        cond_l.append(condition_number(Slw))
        deltas.append(d)
    return cond_s, cond_l, deltas


# ==========================================================================
# 随机矩阵理论（RMT）协方差估计：特征值的“信号/噪声”分离
#
# Marchenko-Pastur(1967) 定理：若 T×n 数据为 i.i.d. 纯噪声（方差 σ²），
# 当 T,n→∞ 且 q=n/T 固定时，样本相关阵的特征值收敛到支撑区间
#     [λ₋, λ₊] = [σ²(1−√q)², σ²(1+√q)²]
# 内的 MP 分布。于是凡落在 λ₊ 以下的特征值"统计上与纯噪声不可区分"，
# 只有超出 λ₊ 的特征值才携带真实相关结构（市场/行业因子）。
#
# 据此得到两个估计量（Bouchaud-Potters 配方，在相关阵上做谱手术再恢复方差）：
#   1. mp_clipped_covariance —— 噪声特征值"压平"为常数（保迹），保留信号；
#   2. pca_factor_covariance —— 只保留 k 个信号特征对 + 对角残差（因子结构）。
#
# 注意：本项目 q = 8/252 ≈ 0.03 很小，噪声带很窄、样本协方差本不算病态，
# RMT 的收益预期有限——这本身就是实验要展示的教学点（RMT 在 n/T 大时才是利器）。
# ==========================================================================
def _cov_to_corr(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """协方差矩阵 → (相关矩阵 C, 标准差向量 s)，谱手术都在 C 上做。"""
    s = np.sqrt(np.diag(S))
    C = S / np.outer(s, s)
    np.fill_diagonal(C, 1.0)
    return C, s


def mp_lambda_bounds(T: int, n: int) -> tuple[float, float]:
    """Marchenko-Pastur 噪声特征值支撑边界 λ± = (1 ∓ √(n/T))²（σ²=1 口径）。"""
    q = n / T
    return float((1.0 - np.sqrt(q)) ** 2), float((1.0 + np.sqrt(q)) ** 2)


def mp_density(lam, q: float, sigma2: float = 1.0) -> np.ndarray:
    """MP 谱密度 ρ(λ) = √((λ₊−λ)(λ−λ₋)) / (2π q σ² λ)，支撑外为 0。绘图用。"""
    lam = np.asarray(lam, dtype=float)
    lo = sigma2 * (1.0 - np.sqrt(q)) ** 2
    hi = sigma2 * (1.0 + np.sqrt(q)) ** 2
    rho = np.zeros_like(lam)
    m = (lam > lo) & (lam < hi)
    rho[m] = np.sqrt((hi - lam[m]) * (lam[m] - lo)) / (2.0 * np.pi * q * sigma2 * lam[m])
    return rho


def mp_clipped_covariance(returns: np.ndarray) -> tuple[np.ndarray, dict]:
    """MP 特征值裁剪协方差（Bouchaud-Potters）。

    步骤：样本协方差 → 相关阵 C → 特征分解 → λᵢ ≤ λ₊ 的"噪声"特征值
    替换为它们的均值（**保迹**：Σλ 不变 = n）→ 重构 → 重新归一使对角恰为 1
    → 乘回样本标准差恢复每资产方差。

    返回 (Sigma_rmt, info)，info 含 k_signal / lambda_plus / eigvals_raw /
    eigvals_clipped（裁剪后、归一前，总和恰为 n，供保迹自检）。
    """
    X = np.asarray(returns, dtype=float)
    T, n = X.shape
    C, s = _cov_to_corr(sample_covariance(X))
    eigval, eigvec = np.linalg.eigh(C)                # 升序
    _, lam_plus = mp_lambda_bounds(T, n)

    noise = eigval <= lam_plus
    lam_clip = eigval.copy()
    if noise.any():
        lam_clip[noise] = eigval[noise].mean()        # 压平噪声、保迹
    C_clip = (eigvec * lam_clip) @ eigvec.T

    d = np.sqrt(np.diag(C_clip))                      # 裁剪轻微破坏对角，归一回 1
    C_clip = C_clip / np.outer(d, d)
    np.fill_diagonal(C_clip, 1.0)

    Sigma = C_clip * np.outer(s, s)
    info = {
        "k_signal": int(np.sum(~noise)),
        "lambda_plus": lam_plus,
        "eigvals_raw": eigval,
        "eigvals_clipped": lam_clip,
    }
    return Sigma, info


def pca_factor_covariance(returns: np.ndarray) -> tuple[np.ndarray, dict]:
    """PCA 因子协方差：只保留 MP 边界以上的 k 个特征对 + 对角特质残差。

        C_f = Σ_{i≤k} λᵢ vᵢvᵢᵀ + D，  D_jj = 1 − Σ_{i≤k} λᵢ v_{ij}²

    D_jj = Σ_{i>k} λᵢ v_{ij}² ≥ 0（数学上非负 ⇒ C_f 半正定），数值上加
    1e-8 下限防 Cholesky 失败；构造使 diag(C_f)=1，乘回标准差恢复方差。

    退化情形：k=0（无信号）→ 纯对角阵 diag(s²)；k=n 不可能——相关阵迹 = n
    ⇒ 平均特征值 = 1 < λ₊，若全部特征值 > λ₊ 则迹 > n·λ₊ > n，矛盾。

    返回 (Sigma_f, info)，info 含 k / lambda_plus / resid_min。
    """
    X = np.asarray(returns, dtype=float)
    T, n = X.shape
    C, s = _cov_to_corr(sample_covariance(X))
    eigval, eigvec = np.linalg.eigh(C)
    _, lam_plus = mp_lambda_bounds(T, n)

    signal = eigval > lam_plus
    k = int(signal.sum())
    assert k < n, "k=n 与迹约束矛盾，不应发生"
    if k == 0:
        return np.diag(s ** 2), {"k": 0, "lambda_plus": lam_plus, "resid_min": 1.0}

    V = eigvec[:, signal]                             # (n, k)
    lam_k = eigval[signal]
    C_f = (V * lam_k) @ V.T
    resid = 1.0 - np.einsum("ij,j,ij->i", V, lam_k, V)
    resid_min = float(resid.min())
    C_f += np.diag(np.maximum(resid, 1e-8))

    Sigma = C_f * np.outer(s, s)
    return Sigma, {"k": k, "lambda_plus": lam_plus, "resid_min": resid_min}
