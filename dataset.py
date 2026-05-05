"""滑动窗口构造监督样本（与论文 Fig.2 一致：输入历史 n 步，输出未来 h 步）。"""
import numpy as np


def create_dataset(data: np.ndarray, seq_len: int, pred_len: int, target_idx: int):
    """
    data: 二维数组，最后一列为缩放后的目标（间接碳排）。
    每行特征去掉当期 carbon，并拼接「窗口末日的 carbon」作为显式历史目标信息。
    """
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len):
        seq = data[i : i + seq_len]
        features = np.delete(seq, target_idx, axis=1)
        last_carbon = float(data[i + seq_len - 1, target_idx])
        last_carbon_col = np.full((seq_len, 1), last_carbon)
        x_seq = np.concatenate([features, last_carbon_col], axis=1)
        X.append(x_seq)
        Y.append(data[i + seq_len : i + seq_len + pred_len, target_idx])
    return np.array(X), np.array(Y)
