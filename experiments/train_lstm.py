"""LSTM 训练：默认 DirectCarbonLSTM；可选论文 Seq2Seq。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import config
from data_pipeline import prepare_supervised
from trainers.torch_trainer import TorchSeqTrainer, TSArrayDataset, min_lr
from utils.calibration import horizon_linear_calibrate
from utils.metrics import per_horizon_metrics, rmse_r2
from utils.plotting import plot_test_suite, plot_training_history


def inverse_targets(arr: np.ndarray, scaler_y):
    flat = arr.reshape(-1, 1)
    inv = scaler_y.inverse_transform(flat)
    return inv.reshape(arr.shape)


def run():
    import os

    if os.environ.get("NEWPROJECT_MAX_EPOCHS"):
        config.LSTM_EPOCHS = int(os.environ["NEWPROJECT_MAX_EPOCHS"])
        config.SHOW_PLOTS = False

    X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler_y = prepare_supervised()
    for name, arr in (
        ("X_train", X_train),
        ("Y_train", Y_train),
        ("X_val", X_val),
        ("Y_val", Y_val),
        ("X_test", X_test),
        ("Y_test", Y_test),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} 含 NaN/Inf，请检查 processed CSV 与特征工程。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_train.shape[2]

    if getattr(config, "LSTM_TYPE", "direct").lower() == "seq2seq":
        from models.lstm_seq2seq import LSTMEncoderDecoder

        model = LSTMEncoderDecoder(
            input_dim=in_dim,
            encoder_hidden=config.LSTM_HIDDEN_DIM,
            decoder_hidden=config.LSTM_HIDDEN_DIM,
            pred_len=config.PRED_LEN,
            num_layers=config.LSTM_SEQ2SEQ_LAYERS,
            dropout=0.0,
        ).to(device)
        tf_ratio = config.LSTM_TEACHER_FORCING
        curve_title = "LSTM Seq2Seq — Training / Validation Loss"
    else:
        from models.direct_lstm import DirectCarbonLSTM

        model = DirectCarbonLSTM(
            input_dim=in_dim,
            hidden_dim=config.LSTM_HIDDEN_DIM,
            pred_len=config.PRED_LEN,
            num_layers=config.LSTM_DIRECT_LAYERS,
            dropout=config.LSTM_DIRECT_DROPOUT,
            bidirectional=True,
            horizon_emb_dim=config.LSTM_HORIZON_EMB,
            n_heads=config.LSTM_DIRECT_HEADS,
            separate_heads=getattr(config, "LSTM_SEPARATE_HEADS", True),
        ).to(device)
        tf_ratio = 0.0
        curve_title = "Direct Carbon LSTM — Training / Validation Loss"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LSTM_LR,
        weight_decay=config.LSTM_WEIGHT_DECAY,
    )

    if config.PRED_LEN > 1:
        from utils.losses import CombinedForecastLoss

        criterion = CombinedForecastLoss(
            config.PRED_LEN,
            start_weight=getattr(config, "LSTM_LOSS_START_WEIGHT", 1.0),
            end_weight=config.LSTM_LOSS_END_WEIGHT,
            smooth_l1_mix=config.LSTM_LOSS_SMOOTH_MIX,
        ).to(device)
    else:
        criterion = nn.MSELoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LSTM_LR_PLateau_FACTOR,
        patience=config.LSTM_LR_PLateau_PATIENCE,
        threshold=config.LSTM_LR_PLateau_THRESHOLD,
        min_lr=config.LSTM_LR_MIN,
    )
    trainer = TorchSeqTrainer(
        model,
        optimizer,
        criterion,
        device,
        scheduler=scheduler,
        max_grad_norm=config.LSTM_MAX_GRAD_NORM,
    )

    train_loader = DataLoader(
        TSArrayDataset(X_train, Y_train),
        batch_size=config.LSTM_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TSArrayDataset(X_val, Y_val), batch_size=config.LSTM_BATCH_SIZE
    )
    test_loader = DataLoader(
        TSArrayDataset(X_test, Y_test), batch_size=config.LSTM_BATCH_SIZE
    )

    best_path = config.CHECKPOINT_LSTM
    best_path.parent.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses, lr_history = [], [], []

    for epoch in range(1, config.LSTM_EPOCHS + 1):
        tr = trainer.train_epoch(train_loader, tf_ratio=tf_ratio)
        va = trainer.evaluate(val_loader, tf_ratio=0.0)
        if not (np.isfinite(tr) and np.isfinite(va)):
            print(
                "\n损失出现 NaN/Inf：多为训练后期梯度爆炸或学习率过大。"
                "已减小默认 LSTM_LR、收紧梯度裁剪；你可继续降低 LSTM_LR、"
                "降低 LSTM_LOSS_END_WEIGHT、或把 LSTM_MAX_GRAD_NORM 调到 0.25。"
            )
            print(f"  在 epoch {epoch} 停止训练；以下使用此前保存的最佳 checkpoint 评估。")
            break
        train_losses.append(tr)
        val_losses.append(va)
        lr_history.append(optimizer.param_groups[0]["lr"])
        trainer.step_scheduler(va)
        min_lr(optimizer, config.LSTM_LR_MIN)
        print(f"Epoch {epoch:03d} | train {tr:.6f} | val {va:.6f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        if trainer.is_best(va, min_delta=config.LSTM_EARLY_STOP_MIN_DELTA):
            trainer.save(str(best_path))
            print(f"  -> 已保存 checkpoint（验证损失新低 {va:.6f}）")

        use_es = config.LSTM_EARLY_STOP and config.LSTM_EARLY_STOP_PATIENCE > 0
        if use_es and trainer.patience_counter >= config.LSTM_EARLY_STOP_PATIENCE:
            print(
                f"早停：验证损失已连续 {config.LSTM_EARLY_STOP_PATIENCE} 个 epoch "
                f"未较历史最佳再下降（min_delta={config.LSTM_EARLY_STOP_MIN_DELTA}）。"
            )
            print(f"  历史最佳 val={trainer.best_val:.6f}，本 epoch val={va:.6f}，共训练 {epoch} 个 epoch。")
            break

    if train_losses:
        plot_training_history(
            train_losses,
            val_losses,
            title=curve_title,
            out_name="lstm_training_curve.png",
            lr_history=lr_history,
        )

    if trainer.best_val == float("inf"):
        raise RuntimeError(
            "未产生任何有效验证损失，无 checkpoint 可加载。请降低 LSTM_LR 或 LSTM_LOSS_END_WEIGHT 后重试。"
        )

    model.load_state_dict(torch.load(str(best_path), map_location=device))
    preds_val, y_val_scaled = trainer.predict(val_loader)
    preds, trues = trainer.predict(test_loader)
    if getattr(config, "VAL_LINEAR_CALIBRATE", False):
        preds = horizon_linear_calibrate(preds_val, y_val_scaled, preds)
    preds_u = inverse_targets(preds, scaler_y)
    trues_u = inverse_targets(trues, scaler_y)

    rmse, r2 = rmse_r2(trues_u, preds_u)
    ph = per_horizon_metrics(preds_u, trues_u)
    tag = "Seq2Seq" if getattr(config, "LSTM_TYPE", "").lower() == "seq2seq" else "Direct"
    min_r2_h = min(h[1] for h in ph) if ph else r2
    print(f"\n[LSTM {tag}] Test RMSE={rmse:.5f}  R2(flat)={r2:.5f}  R2(min over horizons)={min_r2_h:.5f}")
    for i, (r, r2h) in enumerate(ph, start=1):
        print(f"  horizon {i}: RMSE={r:.5f}  R2={r2h:.5f}")

    plot_test_suite(preds_u, trues_u, ph, model_prefix="lstm")

    return {
        "name": f"LSTM_{tag}",
        "rmse": rmse,
        "r2": r2,
        "per_horizon": ph,
    }


if __name__ == "__main__":
    run()
