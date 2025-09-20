#!/usr/bin/env python3
"""Train FT-Transformer on SECweekly features (pure PyTorch, multi-GPU ready).

Usage examples:
  torchrun --nproc_per_node=4 train_ftt.py --db SECweekly.duckb --epochs 50 --batch-size 1024
  python train_ftt.py --db saas_fundamentals.db --epochs 20 --batch-size 512
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from secweekly_data import load_secweekly_dataframe, prepare_supervised_arrays, chronological_split
from ft_transformer import FTTransformer, kd_regression_loss, KDLossConfig


def setup_ddp() -> tuple[bool, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, local_rank, dist.get_world_size()
    return False, 0, 1


def main() -> None:
    ap = argparse.ArgumentParser(description="FT-Transformer training over SECweekly")
    ap.add_argument("--db", required=True, help="Path to SEC DB: .duckdb or .db")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-token", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.7, help="KD alpha weight for true vs teacher")
    ap.add_argument("--save", default="ftt_checkpoint.pt")
    args = ap.parse_args()

    is_ddp, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    df = load_secweekly_dataframe(args.db)
    train_df, val_df = chronological_split(df, frac_val=0.2)
    train = prepare_supervised_arrays(train_df)
    val = prepare_supervised_arrays(val_df)

    Xtr = torch.from_numpy(train.features).to(device)
    Xva = torch.from_numpy(val.features).to(device)
    # Student learns to approximate P/S as proxy if available (y_ps_validation)
    ytr = torch.from_numpy(np.nan_to_num(train.y_ps_validation, nan=np.nan)).to(device)
    yva = torch.from_numpy(np.nan_to_num(val.y_ps_validation, nan=np.nan)).to(device)

    ds_tr = TensorDataset(Xtr, ytr)
    ds_va = TensorDataset(Xva, yva)
    sampler = DistributedSampler(ds_tr) if is_ddp else None
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = FTTransformer(num_features=Xtr.shape[1], d_token=args.d_token, n_blocks=args.blocks, n_heads=args.heads, dropout=args.dropout).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    kd_cfg = KDLossConfig(alpha=args.alpha, temperature=1.0)

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for xb, yb in dl_va:
                pred = model(xb)
                loss = kd_regression_loss(pred, yb, None, kd_cfg)
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("inf")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        if is_ddp:
            sampler.set_epoch(epoch)
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(xb)
                loss = kd_regression_loss(pred, yb, None, kd_cfg)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        val_loss = evaluate()
        if (not is_ddp) or dist.get_rank() == 0:
            print(f"epoch {epoch} val_loss {val_loss:.5f}")
            if val_loss < best:
                best = val_loss
                torch.save({
                    'model_state': model.state_dict() if not is_ddp else model.module.state_dict(),
                    'feature_names': train.feature_names,
                    'scaler_mean': train.scaler_mean,
                    'scaler_std': train.scaler_std,
                }, args.save)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


