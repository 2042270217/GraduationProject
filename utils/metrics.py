import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def rmse_r2(y_true: np.ndarray, y_pred: np.ndarray):
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    return rmse, r2


def per_horizon_metrics(preds: np.ndarray, trues: np.ndarray):
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
        trues = trues.reshape(-1, 1)
    out = []
    for h in range(preds.shape[1]):
        r, r2 = rmse_r2(trues[:, h], preds[:, h])
        out.append((r, r2))
    return out
