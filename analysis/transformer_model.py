"""Small Transformer Encoder for time-series aim trajectory classification."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    n_channels: int
    n_classes: int
    seq_len: int = 200
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    pooling: str = "cls"
    conv_kernel: int = 0
    conv_stride: int = 1


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class AimTransformerEncoder(nn.Module):
    """Encoder-only Transformer over ``[batch, seq_len, n_channels]`` inputs."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        if cfg.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.cfg = cfg

        if cfg.conv_kernel and cfg.conv_kernel > 1:
            self.stem = nn.Sequential(
                nn.Conv1d(cfg.n_channels, cfg.d_model, cfg.conv_kernel, stride=cfg.conv_stride, padding=cfg.conv_kernel // 2),
                nn.GELU(),
            )
            self.input_proj = None
        else:
            self.stem = None
            self.input_proj = nn.Linear(cfg.n_channels, cfg.d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model)) if cfg.pooling == "cls" else None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos = SinusoidalPositionalEncoding(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [batch, seq_len, channels], got {tuple(x.shape)}")
        if self.stem is not None:
            h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        else:
            h = self.input_proj(x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)
        h = self.encoder(self.pos(h))
        if self.cfg.pooling == "cls":
            pooled = h[:, 0]
        else:
            pooled = h.mean(dim=1)
        return self.head(self.norm(pooled))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
