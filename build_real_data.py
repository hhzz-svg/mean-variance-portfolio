# -*- coding: utf-8 -*-
"""
真实数据下载与构建（实验七：真实数据验证用）。

从 Yahoo Finance 的公开行情 JSON 接口下载 8 只美股行业 ETF 的日线复权收盘价，
对齐公共交易日后转成日简单收益率，写入 data/returns_real.csv（与合成数据同结构）。
这 8 只 ETF 正好对应合成数据"8 资产 / 多行业"的设定，便于把六幕结论拿到真实市场上验证。

需要联网。原始 JSON 落在 data/real_raw/（已 gitignore）；处理后的 returns_real.csv
提交入库，保证实验离线可复现。重新生成：  python build_real_data.py

数据来源：query1.finance.yahoo.com/v8/finance/chart（公开接口，无需登录）。
口径：adjclose（含拆分/分红调整）→ 简单日收益 pct_change。区间 2012-01 ~ 2023-12。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "real_raw")
OUT_CSV = os.path.join(HERE, "data", "returns_real.csv")

# 8 只 SPDR 行业 ETF（对应合成数据的多行业结构）
ETFS = [("XLK", "科技"), ("XLF", "金融"), ("XLE", "能源"), ("XLV", "医疗"),
        ("XLP", "必需消费"), ("XLY", "可选消费"), ("XLI", "工业"), ("XLU", "公用")]
PERIOD1, PERIOD2 = 1325376000, 1704067200          # 2012-01-01 ~ 2024-01-01


def download():
    """下载原始 JSON 到 data/real_raw/（需联网）。已存在则跳过。"""
    import urllib.request
    os.makedirs(RAW_DIR, exist_ok=True)
    for t, _ in ETFS:
        path = os.path.join(RAW_DIR, f"{t}.json")
        if os.path.exists(path):
            continue
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
               f"?period1={PERIOD1}&period2={PERIOD2}&interval=1d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            open(path, "wb").write(r.read())
        print(f"下载 {t}")


def build():
    """解析原始 JSON → 对齐公共交易日 → 日简单收益率 → returns_real.csv。"""
    series = {}
    for t, _ in ETFS:
        d = json.load(open(os.path.join(RAW_DIR, f"{t}.json"), encoding="utf-8"))
        r = d["chart"]["result"][0]
        idx = pd.to_datetime(r["timestamp"], unit="s").normalize()
        adj = r["indicators"]["adjclose"][0]["adjclose"]
        s = pd.Series(adj, index=idx).dropna()
        series[t] = s[~s.index.duplicated(keep="last")]

    px = pd.DataFrame(series).dropna()                 # 仅保留全部 ETF 都有的交易日
    rets = px.pct_change().dropna()                    # 日简单收益（与项目口径一致）
    rets = rets.reset_index(drop=True)                 # 用整数索引，与 returns.csv 同结构
    rets.index.name = "day"
    rets.to_csv(OUT_CSV)
    print(f"已写入 {OUT_CSV}: {rets.shape[0]} 天 × {rets.shape[1]} 只 ETF")
    return rets


def asset_sectors() -> dict:
    return {t: sec for t, sec in ETFS}


if __name__ == "__main__":
    download()
    build()
