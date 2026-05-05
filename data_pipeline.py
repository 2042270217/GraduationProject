"""CSV 读取、可选特征增强、缩放与监督样本划分。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

import config
from dataset import create_dataset


def load_processed_df(path=None):
    path = path or config.PROCESSED_CSV
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c == config.TARGET_COL:
            continue
        if c in config.EXCLUDE_FROM_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_scaled_matrix(df: pd.DataFrame, feature_cols: list[str]):
    split_row = max(int(len(df) * config.TRAINVAL_FRAC), 1)
    if getattr(config, "FEATURE_SCALER", "standard").lower() == "robust":
        scaler_x = RobustScaler()
    else:
        scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(df[feature_cols].iloc[:split_row])
    scaler_y.fit(df[[config.TARGET_COL]].iloc[:split_row])

    x_all = scaler_x.transform(df[feature_cols])
    y_all = scaler_y.transform(df[[config.TARGET_COL]])
    full = np.concatenate([x_all, y_all], axis=1)
    target_idx = full.shape[1] - 1
    return full, target_idx, scaler_x, scaler_y, feature_cols


def supervised_splits(full: np.ndarray, target_idx: int):
    seq_len, pred_len = config.SEQ_LEN, config.PRED_LEN
    X, Y = create_dataset(full, seq_len, pred_len, target_idx)

    n_tv = int(len(X) * config.TRAINVAL_FRAC)
    X_tv, Y_tv = X[:n_tv], Y[:n_tv]
    X_test, Y_test = X[n_tv:], Y[n_tv:]

    # 时间顺序：train 取较早窗口，val 取较晚窗口（与测试同为「未来」分布，常优于随机打乱）
    if getattr(config, "TEMPORAL_TRAIN_VAL_SPLIT", True):
        n_train = int(len(X_tv) * config.TRAIN_IN_TRAINVAL_FRAC)
        X_train, Y_train = X_tv[:n_train], Y_tv[:n_train]
        X_val, Y_val = X_tv[n_train:], Y_tv[n_train:]
    else:
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(X_tv))
        X_tv, Y_tv = X_tv[perm], Y_tv[perm]
        n_train = int(len(X_tv) * config.TRAIN_IN_TRAINVAL_FRAC)
        X_train, Y_train = X_tv[:n_train], Y_tv[:n_train]
        X_val, Y_val = X_tv[n_train:], Y_tv[n_train:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def prepare_supervised():
    """
    可选特征工程 + 缩放 + 划分；返回训练/验证/测试张量与目标 StandardScaler（用于反变换）。
    """
    df = load_processed_df()
    if getattr(config, "USE_FEATURE_ENGINEERING", False):
        from features.engineering import drop_redundant_features, enhance_dataframe

        df = enhance_dataframe(df, config.TARGET_COL)
        feats = feature_columns(df)
        thr = getattr(config, "FEATURE_CORR_THRESHOLD", 1.0)
        if thr < 1.0:
            feats = drop_redundant_features(df, feats, thr)
    else:
        feats = feature_columns(df)
    full, target_idx, _, scaler_y, _ = build_scaled_matrix(df, feats)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = supervised_splits(
        full, target_idx
    )
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler_y
