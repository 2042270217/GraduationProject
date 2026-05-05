"""
论文 Fig.3：Encoder LSTM 编码长度为 n 的历史序列；
Decoder LSTM 以 Encoder 最终 (h, c) 初始化，逐步展开长度 h，
各步经全连接层输出间接碳排预测（一步输出整个预测域时可逐时刻回归）。
参考：Seshan et al. (2025) Water Research — LSTM encoder-decoder for N2O；
此处目标变量为污水处理厂间接碳排（carbon）。
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn


class LSTMEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden: int,
        decoder_hidden: int,
        pred_len: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_dim,
            encoder_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 论文：Decoder 另一层 LSTM；逐步输入标量（上一步预测或真值）
        self.decoder = nn.LSTM(
            1,
            decoder_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 若 encoder/decoder hidden 不同，对齐 Decoder 初始状态
        self.enc_to_dec_h = (
            nn.Linear(encoder_hidden, decoder_hidden)
            if encoder_hidden != decoder_hidden
            else nn.Identity()
        )
        self.enc_to_dec_c = (
            nn.Linear(encoder_hidden, decoder_hidden)
            if encoder_hidden != decoder_hidden
            else nn.Identity()
        )
        self.fc_out = nn.Linear(decoder_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        x: [B, n, F]
        y_target: [B, h] 缩放空间真值，用于 teacher forcing
        """
        _, (h_enc, c_enc) = self.encoder(x)
        h0 = self.enc_to_dec_h(h_enc)
        c0 = self.enc_to_dec_c(c_enc)

        b = x.size(0)
        device = x.device
        dec_in = torch.zeros(b, 1, 1, device=device)
        hidden = (h0, c0)
        outs = []

        for t in range(self.pred_len):
            out, hidden = self.decoder(dec_in, hidden)
            pred = self.fc_out(out.squeeze(1))
            outs.append(pred.unsqueeze(1))
            if y_target is not None and random.random() < teacher_forcing_ratio:
                dec_in = y_target[:, t].unsqueeze(-1).unsqueeze(1)
            else:
                dec_in = pred.unsqueeze(1)

        return torch.cat(outs, dim=1).squeeze(-1)
