import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TSArrayDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TorchSeqTrainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None, max_grad_norm=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.best_val = float("inf")
        self.patience_counter = 0
        self.max_grad_norm = max_grad_norm

    def train_epoch(self, loader, tf_ratio: float):
        self.model.train()
        total = 0.0
        n_ok = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x, y, teacher_forcing_ratio=tf_ratio)
            loss = self.criterion(out, y)
            # 非有限 loss 通常是上一轮权重已发散；跳过 step 可避免把 NaN 写回参数
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            bad_grad = False
            for p in self.model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                self.optimizer.zero_grad()
                continue
            self.optimizer.step()
            total += loss.item()
            n_ok += 1
        return total / max(n_ok, 1) if n_ok else float("nan")

    @torch.no_grad()
    def evaluate(self, loader, tf_ratio: float = 0.0):
        self.model.eval()
        total = 0.0
        n_ok = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x, None, teacher_forcing_ratio=tf_ratio)
            loss = self.criterion(out, y)
            if not torch.isfinite(loss):
                return float("nan")
            total += loss.item()
            n_ok += 1
        return total / max(n_ok, 1)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        preds, trues = [], []
        for x, y in loader:
            x = x.to(self.device)
            out = self.model(x, None, teacher_forcing_ratio=0.0)
            preds.append(out.cpu().numpy())
            trues.append(y.numpy())
        return np.concatenate(preds), np.concatenate(trues)

    def step_scheduler(self, val_loss):
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def is_best(self, val_loss: float, min_delta: float = 0.0) -> bool:
        """若 val 较历史最佳提升超过 min_delta，则更新 best 并清零耐心计数；否则累计未提升次数。"""
        if val_loss < self.best_val - min_delta:
            self.best_val = val_loss
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False

    def save(self, path):
        torch.save(self.model.state_dict(), path)


def min_lr(optimizer, floor: float):
    for g in optimizer.param_groups:
        if g["lr"] < floor:
            g["lr"] = floor
