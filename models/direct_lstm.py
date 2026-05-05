"""
直接多步预测：BiLSTM + 时间注意力 + 持久性基线；
每个预测步使用独立 MLP 头（避免共享头在远期步上容量不足）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DirectCarbonLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pred_len: int,
        num_layers: int = 2,
        dropout: float = 0.18,
        bidirectional: bool = True,
        horizon_emb_dim: int = 48,
        n_heads: int = 8,
        input_dropout: float = 0.15,
        use_persistence_bias: bool = True,
        separate_heads: bool = True,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.use_persistence_bias = use_persistence_bias
        self.separate_heads = separate_heads
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        self.input_ln = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        enc_dim = hidden_dim * (2 if bidirectional else 1)
        if enc_dim % n_heads != 0:
            raise ValueError(f"enc_dim={enc_dim} 须能被 n_heads={n_heads} 整除")

        self.attn = nn.MultiheadAttention(enc_dim, n_heads, dropout=dropout, batch_first=True)
        self.enc_norm = nn.LayerNorm(enc_dim)
        self.fuse = nn.Sequential(
            nn.Linear(enc_dim * 2, enc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.post_ln = nn.LayerNorm(enc_dim)
        self.horizon_emb = nn.Embedding(pred_len, horizon_emb_dim)
        head_in = enc_dim + horizon_emb_dim
        mid = max(hidden_dim // 2, 112)

        if separate_heads:
            self.head_mlps = nn.ModuleList()
            for _ in range(pred_len):
                self.head_mlps.append(
                    nn.Sequential(
                        nn.Linear(head_in, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, mid),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(mid, 1),
                    )
                )
            self.head = None
        else:
            self.head = nn.Sequential(
                nn.Linear(head_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, mid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid, 1),
            )
            self.head_mlps = None

        self.horizon_bias = nn.Parameter(torch.zeros(pred_len))

    def forward(self, x: torch.Tensor, target=None, teacher_forcing_ratio: float = 0.0):
        persist_base = x[:, -1, -1]
        x = self.input_ln(self.input_dropout(x))
        out, _ = self.lstm(x)
        out = self.enc_norm(out)
        q = out[:, -1:, :]
        ctx, _ = self.attn(q, out, out, need_weights=False)
        ctx = ctx.squeeze(1)
        last = out[:, -1, :]
        fused = self.post_ln(self.fuse(torch.cat([last, ctx], dim=-1)))

        b = fused.size(0)
        device = fused.device

        if self.separate_heads:
            pieces = []
            for h in range(self.pred_len):
                idx = torch.full((b,), h, device=device, dtype=torch.long)
                he = self.horizon_emb(idx)
                inp = torch.cat([fused, he], dim=-1)
                pieces.append(self.head_mlps[h](inp).squeeze(-1))
            delta = torch.stack(pieces, dim=1)
        else:
            h_idx = torch.arange(self.pred_len, device=device)
            h_emb = self.horizon_emb(h_idx)
            fused_rep = fused.unsqueeze(1).expand(b, self.pred_len, -1)
            h_emb_rep = h_emb.unsqueeze(0).expand(b, -1, -1)
            inp = torch.cat([fused_rep, h_emb_rep], dim=-1)
            delta = self.head(inp).squeeze(-1)

        delta = delta + self.horizon_bias
        if self.use_persistence_bias:
            base = persist_base.unsqueeze(1).expand_as(delta)
            return delta + base
        return delta
