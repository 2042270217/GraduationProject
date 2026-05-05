from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedHorizonMSELoss(nn.Module):
    """按预测步加权 MSE。默认从 start_weight 线性插值到 end_weight，长度随 pred_len 自动对齐。"""

    def __init__(
        self,
        pred_len: int,
        start_weight: float = 1.0,
        end_weight: float = 1.55,
        horizon_weights: tuple | list | None = None,
    ):
        super().__init__()
        if horizon_weights is not None:
            if len(horizon_weights) != pred_len:
                raise ValueError("horizon_weights 长度须等于 pred_len")
            w = torch.tensor(horizon_weights, dtype=torch.float32)
        else:
            w = torch.linspace(float(start_weight), float(end_weight), steps=pred_len)
        self.register_buffer("w", w.view(1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        se = (pred - target) ** 2
        w = self.w.to(device=pred.device, dtype=pred.dtype)
        return (se * w).mean()


class CombinedForecastLoss(nn.Module):
    """加权 MSE + Smooth L1。"""

    def __init__(
        self,
        pred_len: int,
        start_weight: float = 1.0,
        end_weight: float = 1.55,
        smooth_l1_mix: float = 0.18,
        horizon_weights: tuple | list | None = None,
    ):
        super().__init__()
        self.wmse = WeightedHorizonMSELoss(
            pred_len,
            start_weight=start_weight,
            end_weight=end_weight,
            horizon_weights=horizon_weights,
        )
        self.smooth_l1_mix = smooth_l1_mix

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        m = self.wmse(pred, target)
        if self.smooth_l1_mix <= 0:
            return m
        s = F.smooth_l1_loss(pred, target)
        return (1.0 - self.smooth_l1_mix) * m + self.smooth_l1_mix * s
