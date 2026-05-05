"""在标准化之前构造滞后/滚动/差分等业务特征，一次性 concat 降低 DataFrame 碎片化。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def enhance_dataframe(df: pd.DataFrame, target_col: str = "carbon") -> pd.DataFrame:
    """
    对碳排预测常用的工艺变量做滞后与滚动统计；目标列也构造滞后（仅过去信息）。
    """
    out = df.copy()
    key_cols = [
        "flow",
        "power",
        "cod_in",
        "tn_in",
        "nh3_in",
        "mlss",
        target_col,
    ]
    new_cols: dict[str, pd.Series] = {}

    for col in key_cols:
        if col not in out.columns:
            continue
        s = out[col].astype(float)
        for lag in (1, 2, 3, 7, 14):
            new_cols[f"{col}_lag{lag}"] = s.shift(lag)
        for w in (7, 14, 21):
            new_cols[f"{col}_roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
            new_cols[f"{col}_roll{w}_std"] = s.rolling(w, min_periods=1).std()
        new_cols[f"{col}_diff1"] = s.diff(1)

    if "power" in out.columns and "flow" in out.columns:
        new_cols["power_per_flow"] = out["power"].astype(float) / (
            out["flow"].astype(float).abs() + 1e-6
        )
    if "chemical" in out.columns and "flow" in out.columns:
        new_cols["chemical_per_flow"] = out["chemical"].astype(float) / (
            out["flow"].astype(float).abs() + 1e-6
        )
    if "cod_in" in out.columns and "cod_out" in out.columns:
        r = out["cod_in"].astype(float) / (out["cod_out"].astype(float).abs() + 1e-6)
        new_cols["cod_in_out_ratio"] = r.clip(0.0, 120.0)

    if new_cols:
        block = pd.DataFrame(new_cols, index=out.index)
        out = pd.concat([out, block], axis=1)

    out = out.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    num = out.select_dtypes(include=[np.number]).columns
    out[num] = out[num].fillna(0.0)
    return out


def drop_redundant_features(df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.995):
    """极高相关列去冗余（保留先出现的列）。"""
    if len(feature_cols) < 2:
        return feature_cols
    sub = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = sub.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {c for c in upper.columns if any(upper[c] > threshold)}
    return [c for c in feature_cols if c not in drop]
