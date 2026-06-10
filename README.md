# 均值-方差投资组合：矩阵算法、解析解与样本外实验

> Markowitz mean–variance portfolio optimization, built from scratch (NumPy-only) and studied from a **matrix-algorithms** viewpoint — then stress-tested out-of-sample.

[![CI](https://github.com/hhzz-svg/mean-variance-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/hhzz-svg/mean-variance-portfolio/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/built%20with-NumPy-013243?logo=numpy&logoColor=white)
![No black-box solver](https://img.shields.io/badge/optimizer-from%20scratch-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Reproducible](https://img.shields.io/badge/seed-fixed%20·%20reproducible-success)

把量化金融最经典的资产配置问题 —— **Markowitz 均值-方差模型** —— 建成一个**双目标规划**
（同时 `max μᵀw` 收益、`min wᵀΣw` 风险），从**矩阵算法**视角完整求解：闭式解、从零实现的
数值优化器、协方差收缩，再用**样本外滚动回测**验证"样本内最优为什么在现实中会翻车"。

![样本外累计净值](figures/backtest_wealth.png)

> **一图看懂**：用真实 μ 的 Oracle 组合（青色）一骑绝尘，但现实中 μ 要靠估计——朴素的等权
> 1/N（灰色）反而跑赢所有"优化"出来的组合。这正是本项目的故事主线。

---

## TL;DR (English)

A from-scratch (NumPy-only) study of the Markowitz mean–variance problem as a bi-objective
program, solved from a matrix-algorithms viewpoint: closed-form efficient frontier via
**Cholesky** factorization, **projected-gradient descent with Euclidean simplex projection**
for the long-only (no-short) case, and **Ledoit–Wolf shrinkage** covariance. A strict
no-look-ahead **walk-forward backtest** then shows why the in-sample optimum fails in
practice — estimation error in `μ` collapses the max-Sharpe portfolio (OOS Sharpe goes
**negative**) while naïve **1/N wins**, reproducing DeMiguel et al. (2009). All solvers are
cross-validated against SciPy and ship with numerical self-checks (run on every push by CI).

```bash
pip install -r requirements.txt
python main.py                   # Act 1 — in-sample: frontier, GMV, tangency + 6 self-checks
python experiment_backtest.py    # Act 2 — out-of-sample walk-forward backtest + 5 self-checks
```

Full math derivation (in Chinese): [docs/derivation.pdf](docs/derivation.pdf).

---

## 两幕故事

### 第一幕 · 样本内：把优化问题手推手写一遍

| 模块 | 内容 | 矩阵算法 / ML 要点 |
|---|---|---|
| 解析法 | 拉格朗日/KKT 闭式解、有效前沿、GMV、切点组合 | **Cholesky** 分解解 SPD 系统、`A/B/C/D` 标量、前代/回代手写 |
| 数值法 | **从零实现**投影梯度下降（处理不可卖空 `w≥0`） | 概率单纯形欧氏投影（KKT 推导）、`O(1/k)` 收敛性 |
| 估计 | 样本 `μ/Σ` + **Ledoit-Wolf 收缩**协方差 | 统计学习正则化、偏差-方差权衡、条件数 |
| 数据 | 三因子模型生成的 8 资产合成日收益率 | 行业块状相关结构，离线、可复现 |

**主要结果**（合成 8 资产，年化口径，`r_f = 3%`）：

| 组合 | 年化收益 | 年化波动 | 夏普比 | 特点 |
|---|---|---|---|---|
| 全局最小方差 GMV | 6.36% | 15.34% | — | 集中于低波动金融/消费股 |
| 切点组合（最大夏普，允许卖空） | 18.54% | 32.97% | **0.471** | 做多高夏普、做空低收益资产 |
| 长仓最小方差（`w≥0`） | 6.63% | 15.38% | 0.236 | 全部非负，约束使前沿内移 |

核心算法**纯 numpy 从零实现**，并通过 6 项数值自检 + 与 **SciPy SLSQP** 的独立交叉验证
（最大权重偏差 `< 5e-6`）。详见 [`results/summary.json`](results/summary.json) 与
中文 LaTeX 数学推导：**[📄 直接看 PDF](docs/derivation.pdf)**（10 页，源文件
[`docs/derivation.tex`](docs/derivation.tex)）。

### 第二幕 · 样本外：估计误差让"最优"翻车

第一幕全是**样本内**的。现实里 `μ/Σ` 都得从历史估计，估计误差会摧毁"样本内最优"。本实验用
walk-forward 协议（估计窗 `L=252` 天、每 `R=21` 天再平衡、**严格无前视**）在 10 年合成序列上
对比 7 个策略：

| 策略 | 样本内夏普 | **样本外夏普** | 年化波动 | 最大回撤 | 平均换手 |
|---|---|---|---|---|---|
| 等权 1/N | 0.39 | **0.45** | 18.4% | −47% | 0% |
| GMV（样本Σ） | 0.23 | 0.18 | 16.1% | −41% | 6.0% |
| GMV（收缩Σ） | 0.23 | 0.20 | 16.1% | −41% | 5.6% |
| 切点（样本μ,Σ，允许卖空） | 0.48 | **−0.17** | 1009% | −3827% | 2429% |
| 切点（收缩Σ） | 0.48 | 0.37 | 483% | −110% | 1881% |
| 长仓最小方差 | 0.25 | 0.22 | 16.1% | −42% | 4.8% |
| Oracle 切点（用真 μ） | 0.48 | **0.49** | 32.5% | −64% | 12.5% |

**结论（经典 DeMiguel et al. 2009）**：样本内夏普最高的切点组合，样本外夏普塌缩到负值、
波动/回撤/换手全部爆炸；朴素的等权 1/N 反而胜过所有需要估计的优化策略。而 **Oracle 切点
（用真实 μ）样本内外都最优** —— 干净地证明问题出在 **μ 的估计误差**，而非方法本身。
Ledoit-Wolf 收缩把切点的样本外夏普从 −0.17 救回 0.37，逐期条件数也始终更良态。

| | |
|---|---|
| ![夏普塌缩](figures/backtest_sharpe_is_vs_oos.png) | ![换手与条件数](figures/backtest_turnover_condition.png) |

> 口径说明：再平衡到目标权重后持有至下期（忽略持有期内权重漂移）；切点策略不设杠杆上限，
> 故意保留其样本外失控以暴露估计误差——其单日亏损常 >100%，净值图已将其剔除（失败由表格量化）。

---

## 如何运行

```bash
# 1. 安装依赖（numpy / matplotlib / pandas；scipy 仅用于交叉验证）
pip install -r requirements.txt

# 2. 第一幕——样本内：生成数据、求解、出 5 张图、跑自检、导出结果
python main.py

# 3. 第二幕——样本外：滚动回测，出 4 张图、导出 backtest_summary.json
python experiment_backtest.py

# 4.（可选）编译中文数学推导 PDF（需 xelatex，运行两遍以生成目录/交叉引用）
cd docs && xelatex derivation.tex && xelatex derivation.tex
```

> Windows 提示：图像中文字体用系统自带的 Microsoft YaHei / SimHei；控制台中文若乱码，
> 脚本已自动把 stdout 切到 UTF-8。全流程随机种子固定（`seed=42`），结果可复现。

## 目录结构

```
.
├── README.md                   本文件
├── LICENSE                     MIT
├── .github/workflows/ci.yml    CI：每次 push 自动跑两幕全部 11 项自检
├── requirements.txt            依赖
├── main.py                     第一幕：数据→估计→解析/数值求解→图像→自检→导出
├── experiment_backtest.py      第二幕：样本外滚动回测 + 自检
├── src/
│   ├── generate_data.py        三因子模型生成合成收益率（含真实 μ）
│   ├── data_utils.py           μ/Σ 估计 + Ledoit-Wolf 收缩 + 条件数
│   ├── analytic.py             解析法（Cholesky、闭式前沿、GMV、切点）
│   ├── numeric.py              数值法（投影梯度下降 + 单纯形投影）
│   ├── backtest.py             样本外滚动回测引擎 + 7 策略注册表
│   ├── metrics.py              收益/风险/夏普 + 回撤/换手/年化统计
│   └── plots.py                全部图像绘制（样本内 5 张 + 回测 4 张）
├── figures/                    9 张输出图（运行后生成）
├── results/
│   ├── summary.json            样本内关键结果 + 自检
│   └── backtest_summary.json   样本外绩效 + 自检
├── data/                       合成日收益率 CSV（运行后生成）
└── docs/
    ├── derivation.tex          中文数学推导（ctex + xelatex）
    └── derivation.pdf          编译产物（10 页）
```

## 交付物对应

1. **数学推导（LaTeX）**：[docs/derivation.tex](docs/derivation.tex) → [docs/derivation.pdf](docs/derivation.pdf)
2. **代码**：[main.py](main.py) + [experiment_backtest.py](experiment_backtest.py) + [src/](src/)（核心算法纯 numpy 从零实现）
3. **图像**：[figures/](figures/) 共 9 张（有效前沿、可行域、PGD 收敛、协方差分析、权重对比 + 回测净值/夏普/换手/滚动权重）
4. **现实问题落地**：从抽象矩阵优化到"如何配一篮子股票"，并用样本外实验解读分散化、做空约束、估计误差的真实代价

## 参考

- Markowitz (1952), *Portfolio Selection*.
- Ledoit & Wolf (2004), *A well-conditioned estimator for large-dimensional covariance matrices*.
- DeMiguel, Garlappi & Uppal (2009), *Optimal Versus Naive Diversification*.

## License

[MIT](LICENSE) © 2026 忽哲
