#!/usr/bin/env python3
"""FT-Transformer (simplified) for tabular regression in pure PyTorch.

Implements column-token embedding + Transformer encoder + regression head.
Suitable for DDP multi-GPU via torchrun.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        # return tokens: [B, F, d]
        return x.unsqueeze(-1) * self.weight + self.bias


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_hidden: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention over feature tokens
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.lin2(F.gelu(self.lin1(x)))
        x = self.norm2(x + self.dropout(ff))
        return x


class FTTransformer(nn.Module):
    def __init__(self, num_features: int, d_token: int = 192, n_blocks: int = 4, n_heads: int = 8, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_token, n_heads=n_heads, d_hidden=d_token * ff_mult, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        tok = self.tokenizer(x)  # [B, F, d]
        h = tok
        for blk in self.blocks:
            h = blk(h)
        # pooling over feature tokens (mean)
        h = h.mean(dim=1)
        out = self.reg_head(h).squeeze(-1)
        return out


@dataclass
class KDLossConfig:
    alpha: float = 0.7  # weight on supervised loss vs teacher
    temperature: float = 1.0


def kd_regression_loss(student_pred: torch.Tensor, y_true: Optional[torch.Tensor], teacher_pred: Optional[torch.Tensor], cfg: KDLossConfig) -> torch.Tensor:
    losses = []
    if y_true is not None:
        mask = torch.isfinite(y_true)
        if mask.any():
            losses.append(F.l1_loss(student_pred[mask], y_true[mask]))
    if teacher_pred is not None:
        mask_t = torch.isfinite(teacher_pred)
        if mask_t.any():
            # temperature is identity for regression; included for API symmetry
            losses.append(F.mse_loss(student_pred[mask_t], teacher_pred[mask_t]))
    if not losses:
        return torch.tensor(0.0, device=student_pred.device)
    if y_true is not None and teacher_pred is not None:
        return cfg.alpha * losses[0] + (1.0 - cfg.alpha) * losses[1]
    return losses[0]


