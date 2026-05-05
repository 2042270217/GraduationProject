"""
依次运行 LSTM（Encoder–Decoder）、XGBoost、GBM，输出测试集对比表。
需在 NewProject 根目录执行：python run_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from experiments.train_lstm import run as run_lstm
    from experiments.train_xgboost import run as run_xgb
    from experiments.train_gbm import run as run_gbm

    rows = []
    rows.append(run_lstm())
    rows.append(run_xgb())
    rows.append(run_gbm())

    print("\n" + "=" * 60)
    print("多模型对比（测试集，原始量纲碳排）")
    print("=" * 60)
    print(f"{'Model':<14} {'RMSE':>12} {'R2':>10}")
    print("-" * 60)
    for r in rows:
        print(f"{r['name']:<14} {r['rmse']:12.5f} {r['r2']:10.5f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
