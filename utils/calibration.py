"""在验证集上按预测步做线性校准 y ≈ a·ŷ + b，再用于测试预测（缓解系统性偏差）。"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def horizon_linear_calibrate(
    preds_val: np.ndarray,
    y_val: np.ndarray,
    preds_test: np.ndarray,
) -> np.ndarray:
    preds_val = np.asarray(preds_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    preds_test = np.asarray(preds_test, dtype=np.float64).copy()
    if preds_val.ndim == 1:
        preds_val = preds_val.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        preds_test = preds_test.reshape(-1, 1)
    h = preds_val.shape[1]
    out = preds_test.copy()
    for j in range(h):
        reg = LinearRegression()
        reg.fit(preds_val[:, j].reshape(-1, 1), y_val[:, j])
        out[:, j] = reg.predict(preds_test[:, j].reshape(-1, 1))
    return out.astype(np.float32)
