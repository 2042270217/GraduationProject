"""
超参数：针对 7 日预测拉高远期步表现 — 独立 horizon 头 + 远期加权损失 + 更大编码器。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PLOT_DIR = PROJECT_ROOT / "figures"
SHOW_PLOTS = True
PROCESSED_CSV = PROJECT_ROOT / "processed" / "processed_data.csv"
CHECKPOINT_LSTM = PROJECT_ROOT / "checkpoints" / "lstm_seq2seq.pt"
CHECKPOINT_XGB = PROJECT_ROOT / "checkpoints" / "xgboost_model.pkl"
CHECKPOINT_GBM = PROJECT_ROOT / "checkpoints" / "gbm_model.pkl"

TARGET_COL = "carbon"
EXCLUDE_FROM_FEATURES = ("date", "month")

SEQ_LEN = 56
PRED_LEN = 3

USE_FEATURE_ENGINEERING = True
FEATURE_CORR_THRESHOLD = 0.99
FEATURE_SCALER = "robust"

LSTM_TYPE = "direct"
LSTM_SEPARATE_HEADS = True

LSTM_HIDDEN_DIM = 384
LSTM_SEQ2SEQ_LAYERS = 1
LSTM_DIRECT_LAYERS = 3
LSTM_DIRECT_DROPOUT = 0.3
LSTM_DIRECT_HEADS = 8
LSTM_HORIZON_EMB = 64

LSTM_BATCH_SIZE = 48
LSTM_EPOCHS = 260
LSTM_LR = 1.5e-4
LSTM_LR_MIN = 1e-6
LSTM_WEIGHT_DECAY = 1e-4
LSTM_LR_PLateau_FACTOR = 0.5
LSTM_LR_PLateau_PATIENCE = 8
LSTM_LR_PLateau_THRESHOLD = 3e-3
LSTM_TEACHER_FORCING = 0.5
LSTM_MAX_GRAD_NORM = 0.5

LSTM_EARLY_STOP = True
LSTM_EARLY_STOP_PATIENCE = 35
LSTM_EARLY_STOP_MIN_DELTA = 1e-5

# 按步加权损失：自动随 PRED_LEN 生成权重（D+1→D+H 从首到尾线性插值），无需手写元组
LSTM_LOSS_START_WEIGHT = 1.0
LSTM_LOSS_END_WEIGHT = 2.5
LSTM_LOSS_SMOOTH_MIX = 0.14

TRAINVAL_FRAC = 0.8
TRAIN_IN_TRAINVAL_FRAC = 0.85
# True：train/val 按时间先后切（推荐）；False：对样本随机打乱后再切（论文写法）
TEMPORAL_TRAIN_VAL_SPLIT = True
# 在验证集上按步线性校准后再评估测试集（提升 R²，部署时需在同类数据上保留校准步骤）
VAL_LINEAR_CALIBRATE = True

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1,
)
GBM_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.04,
    subsample=0.85,
    random_state=42,
)
