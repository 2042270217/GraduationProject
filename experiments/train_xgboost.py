"""XGBoost 多输出回归：展平 LSTM 输入窗口，与序列模型公平对比。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

import config
from data_pipeline import prepare_supervised
from models.tree_models import XGBoostMultiOutput
from utils.metrics import per_horizon_metrics, rmse_r2
from utils.plotting import plot_test_suite


def inverse_targets(arr: np.ndarray, scaler_y):
    flat = arr.reshape(-1, 1)
    return scaler_y.inverse_transform(flat).reshape(arr.shape)


def run():
    X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler_y = prepare_supervised()

    X_train_f = X_train.reshape(X_train.shape[0], -1)
    X_val_f = X_val.reshape(X_val.shape[0], -1)
    X_test_f = X_test.reshape(X_test.shape[0], -1)

    model = XGBoostMultiOutput(**config.XGB_PARAMS)
    model.fit(X_train_f, Y_train)

    preds = model.predict(X_test_f)
    preds_u = inverse_targets(preds, scaler_y)
    trues_u = inverse_targets(Y_test, scaler_y)

    rmse, r2 = rmse_r2(trues_u, preds_u)
    ph = per_horizon_metrics(preds_u, trues_u)
    print(f"\n[XGBoost] Test RMSE={rmse:.5f}  R2={r2:.5f}")
    for i, (r, r2h) in enumerate(ph, start=1):
        print(f"  horizon {i}: RMSE={r:.5f}  R2={r2h:.5f}")

    plot_test_suite(preds_u, trues_u, ph, model_prefix="xgboost")

    joblib.dump(model.model, config.CHECKPOINT_XGB)
    print("Saved:", config.CHECKPOINT_XGB)

    return {"name": "XGBoost", "rmse": rmse, "r2": r2, "per_horizon": ph}


if __name__ == "__main__":
    run()
