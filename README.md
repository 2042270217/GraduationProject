# 污水处理厂间接碳排预测 — LSTM / XGBoost / GBM

本项目基于 **Seshan et al. (2025)** *Water Research* 中 **LSTM Encoder–Decoder** 的数据处理与建模思路（线性插值、滚动均值、序列监督样本、Z-score 仅用 train+val 段估计量；模型结构为 Encoder → Decoder 逐步解码），目标变量改为报表中的 **间接碳排放（carbon）**，并与 **XGBoost**、**梯度提升树（GBM）** 在同一滑动窗口特征下进行对比。

## 环境与数据

```bash
cd NewProject
pip install -r requirements.txt
```

将各月月报 `.xls` 置于 `data/`（与本仓库一致的表格结构）。生成预处理 CSV：

```bash
python data_preprocessing.py
```

## 训练与对比

```bash
python experiments/train_lstm.py
python experiments/train_xgboost.py
python experiments/train_gbm.py
```

一键顺序运行三模型并打印测试集 RMSE / R²：

```bash
python run_comparison.py
```

训练与测试图默认保存到 **`figures/`**（训练曲线含 **log 尺度损失** 与 **学习率**；测试集含 **时序对比**、**真值–预测散点**、**残差图**；多步时另有 **各步 RMSE/R² 柱状图**）。无图形界面时在 `config.py` 设 `SHOW_PLOTS = False` 仅保存 PNG。

主要超参数见 `config.py`（如 Seq2Seq 256–256、`SEQ_LEN`/`PRED_LEN`、论文对齐 batch=32、AdamW+MSE 等）。

## 参考文献

Seshan, S., et al. (2025). Forecasting nitrous oxide emissions from a full-scale wastewater treatment plant using LSTM-based deep learning models. *Water Research*, 268, 122754.

（方法论对齐此文；预测对象为碳排而非 N₂O，数据集为你的报表字段。）
