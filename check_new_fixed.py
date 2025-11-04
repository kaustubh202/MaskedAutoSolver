# check_new_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
import time
import logging
import sys
import argparse
import json
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--num_views", type=int, default=4, help="N = number of masks/views")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--outdir", type=str, default="grid_results", help="Where to write all outputs")
parser.add_argument("--epochs", type=int, default=1500, help="Max pretraining epochs")
parser.add_argument("--data_path", type=str, default=None, help="Path to dataset .npz")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for pretraining")
parser.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio for pretraining")
parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
parser.add_argument("--memory_efficient", action="store_true", help="Use sequential view processing")
parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
parser.add_argument("--auto_adjust_batch", action="store_true", help="Auto-adjust batch size based on num_views")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.outdir, "training_log.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_losses(train_mse, val_mse, title="Loss Curve", save_path="loss_curve.png"):
    plt.figure(figsize=(12, 7))
    plt.plot(train_mse, label="Train masked MSE", linewidth=2)
    plt.plot(val_mse, label="Val masked MSE", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Log Scale)", fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Loss curve saved to {save_path}")

class PretrainingDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data.astype(np.float32))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

def make_multi_view_collate_maskV(mask_ratio=0.75, num_views=2):
    def collate_fn(batch):
        batch = torch.stack(batch, dim=0)
        B, nb, td = batch.shape
        masks = torch.zeros((num_views, B, nb), dtype=torch.bool)
        num_to_mask = max(1, int(math.floor(mask_ratio * nb)))
        for v in range(num_views):
            for i in range(B):
                idxs = np.random.choice(nb, size=num_to_mask, replace=False)
                masks[v, i, idxs] = True
        mask_feat = torch.zeros((num_views, B, nb, td), dtype=torch.bool)
        mask_feat[..., -2:] = masks.unsqueeze(-1).expand(-1, -1, -1, 2)
        return {"tokens": batch, "masks": masks, "mask_feat": mask_feat, "targets": batch.clone()}
    return collate_fn

def train_one_epoch_sequential(model, dataloader, optimizer, device):
    """Memory-efficient sequential view processing."""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        mask_feat = batch["mask_feat"].to(device)
        targets = batch["targets"].to(device)
        K, B, nb, td = mask_feat.shape
        
        optimizer.zero_grad()
        loss_sum = 0.0
        
        for v in range(K):
            masked_tokens = tokens.clone()
            masked_tokens = masked_tokens.masked_fill_(mask_feat[v], 0.0)
            preds = model(masked_tokens)
            
            mf = mask_feat[v]
            mse_loss = ((preds - targets)**2) * mf.float()
            denom = mf.float().sum() + 1e-8
            view_loss = mse_loss.sum() / denom
            
            # Backward on each view separately
            (view_loss / K).backward()
            loss_sum += view_loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss_sum / K
    
    return total_loss / len(dataloader)

def train_one_epoch_accumulation(model, dataloader, optimizer, device, accumulation_steps=2):
    """Gradient accumulation for better memory/speed tradeoff."""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        mask_feat = batch["mask_feat"].to(device)
        targets = batch["targets"].to(device)
        K, B, nb, td = mask_feat.shape
        
        optimizer.zero_grad()
        views_per_chunk = max(1, math.ceil(K / accumulation_steps))
        total_loss_batch = 0.0
        
        for chunk_idx in range(accumulation_steps):
            start_v = chunk_idx * views_per_chunk
            end_v = min(start_v + views_per_chunk, K)
            if start_v >= K:
                break
            
            masked_list = []
            for v in range(start_v, end_v):
                masked = tokens.clone().masked_fill_(mask_feat[v], 0.0)
                masked_list.append(masked)
            
            chunk_batch = torch.cat(masked_list, dim=0)
            preds_chunk = model(chunk_batch).view(end_v - start_v, B, nb, td)
            
            loss_sum = 0.0
            for idx, v in enumerate(range(start_v, end_v)):
                mf = mask_feat[v]
                mse_loss = ((preds_chunk[idx] - targets)**2) * mf.float()
                denom = mf.float().sum() + 1e-8
                loss_sum += mse_loss.sum() / denom
            
            chunk_loss = loss_sum / float(K)
            chunk_loss.backward()
            total_loss_batch += loss_sum.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += total_loss_batch / K
    
    return total_loss / len(dataloader)

def train_one_epoch_packed(model, dataloader, optimizer, device):
    """Original packed version - fast but memory-intensive."""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        mask_feat = batch["mask_feat"].to(device)
        targets = batch["targets"].to(device)
        K, B, nb, td = mask_feat.shape
        
        masked_list = [tokens.clone().masked_fill_(mask_feat[v], 0.0) for v in range(K)]
        big_batch = torch.cat(masked_list, dim=0)
        
        optimizer.zero_grad()
        preds_all = model(big_batch).view(K, B, nb, td)
        
        loss_sum = 0.0
        for v in range(K):
            mf = mask_feat[v]
            mse_loss = ((preds_all[v] - targets)**2) * mf.float()
            denom = mf.float().sum() + 1e-8
            loss_sum += mse_loss.sum() / denom
        
        loss_avg = loss_sum / float(K)
        loss_avg.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss_avg.item()
    
    return total_loss / len(dataloader)

def validate_maskV(model, dataloader, device, memory_efficient=False):
    """Validation with memory-efficient option."""
    model.eval()
    total_mse, total_mae = 0.0, 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            mask_feat = batch["mask_feat"].to(device)
            targets = batch["targets"].to(device)
            K, B, nb, td = mask_feat.shape
            
            if memory_efficient:
                # Sequential processing
                batch_mse, batch_mae = 0.0, 0.0
                for v in range(K):
                    masked = tokens.clone().masked_fill_(mask_feat[v], 0.0)
                    preds = model(masked)
                    
                    mf = mask_feat[v]
                    denom = mf.float().sum() + 1e-8
                    batch_mse += (((preds - targets)**2) * mf.float()).sum().item() / denom.item()
                    batch_mae += (torch.abs(preds - targets) * mf.float()).sum().item() / denom.item()
            else:
                # Packed processing
                masked_list = [tokens.clone().masked_fill_(mask_feat[v], 0.0) for v in range(K)]
                big_batch = torch.cat(masked_list, dim=0)
                preds_all = model(big_batch).view(K, B, nb, td)
                
                batch_mse, batch_mae = 0.0, 0.0
                for v in range(K):
                    mf = mask_feat[v]
                    denom = mf.float().sum() + 1e-8
                    batch_mse += (((preds_all[v] - targets)**2) * mf.float()).sum().item() / denom.item()
                    batch_mae += (torch.abs(preds_all[v] - targets) * mf.float()).sum().item() / denom.item()
            
            total_mse += batch_mse / float(K)
            total_mae += batch_mae / float(K)
    
    return total_mse / len(dataloader), total_mae / len(dataloader)

def get_dataloaders(npz_path, pretrain_frac=0.9, batch_size_pre=128, mask_ratio=0.75, 
                    num_views=2, seed=42):
    data = np.load(npz_path)
    full_tokens = np.concatenate([data['X_tokens'], data['Y_targets']], axis=-1).astype(np.float32)
    
    num_samples = full_tokens.shape[0]
    indices = np.arange(num_samples)
    np.random.RandomState(seed).shuffle(indices)
    
    num_pretrain_pool = int(pretrain_frac * num_samples)
    pretrain_pool_indices = indices[:num_pretrain_pool]
    
    means = full_tokens[pretrain_pool_indices].mean(axis=(0, 1))
    stds = full_tokens[pretrain_pool_indices].std(axis=(0, 1)) + 1e-8
    full_tokens = (full_tokens - means[None, None, :]) / stds[None, None, :]
    
    num_pretrain_train = int(0.9 * len(pretrain_pool_indices))
    pretrain_train_indices = pretrain_pool_indices[:num_pretrain_train]
    pretrain_val_indices = pretrain_pool_indices[num_pretrain_train:]
    
    pretrain_train_ds = PretrainingDataset(full_tokens[pretrain_train_indices])
    pretrain_val_ds = PretrainingDataset(full_tokens[pretrain_val_indices])
    
    collate_fn = make_multi_view_collate_maskV(mask_ratio=mask_ratio, num_views=num_views)
    pretrain_loader = DataLoader(pretrain_train_ds, batch_size=batch_size_pre, 
                                 shuffle=True, collate_fn=collate_fn)
    pretrain_val_loader = DataLoader(pretrain_val_ds, batch_size=batch_size_pre, 
                                     shuffle=False, collate_fn=collate_fn)
    
    metadata = {"means": means, "stds": stds}
    return {"pretrain_loader": pretrain_loader, "pretrain_val_loader": pretrain_val_loader, 
            "metadata": metadata}

class MAEModel(nn.Module):
    def __init__(self, token_dim, embed_dim, num_buses, num_layers=8, num_heads=8):
        super().__init__()
        self.token_embedding = nn.Linear(token_dim, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_buses, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(embed_dim, token_dim)
    
    def forward(self, x):
        x = self.token_embedding(x) + self.positional_embedding
        x = self.encoder(x)
        return self.decoder(x)

def pretrain_main(dataloaders, token_dim, num_buses, outdir, embed_dim=64, num_layers=4,
                  epochs=200, lr=1e-4, weight_decay=1e-5, patience=25, device="cpu",
                  memory_efficient=False, accumulation_steps=2):
    
    model = MAEModel(token_dim, embed_dim, num_buses, num_layers=num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                            factor=0.5, patience=10)
    
    # Choose training function based on memory mode
    if memory_efficient:
        train_fn = train_one_epoch_sequential
        logging.info("Using SEQUENTIAL view processing (memory-efficient)")
    elif accumulation_steps > 1:
        train_fn = lambda m, d, o, dev: train_one_epoch_accumulation(m, d, o, dev, accumulation_steps)
        logging.info(f"Using GRADIENT ACCUMULATION with {accumulation_steps} steps")
    else:
        train_fn = train_one_epoch_packed
        logging.info("Using PACKED view processing (fastest, memory-intensive)")
    
    best_val, patience_count = float('inf'), 0
    training_losses, val_mse_losses, val_mae_losses = [], [], []
    
    logging.info("--- Starting Pre-training ---")
    training_start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_fn(model, dataloaders['pretrain_loader'], optimizer, device)
        val_mse, val_mae = validate_maskV(model, dataloaders['pretrain_val_loader'], 
                                         device, memory_efficient=memory_efficient)
        
        training_losses.append(train_loss)
        val_mse_losses.append(val_mse)
        val_mae_losses.append(val_mae)
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch+1:03d}/{epochs} | Train: {train_loss:.6e} | "
                        f"Val MSE: {val_mse:.6e} | Val MAE: {val_mae:.6e}")
        
        scheduler.step(val_mse)
        
        if val_mse < best_val:
            best_val = val_mse
            patience_count = 0
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                       "epoch": epoch}, os.path.join(outdir, "mae_best.pt"))
        else:
            patience_count += 1
            if patience_count >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    total_training_time = time.time() - training_start_time
    logging.info(f"Total training time: {total_training_time // 60:.0f}m {total_training_time % 60:.2f}s")
    
    plot_losses(training_losses, val_mse_losses, "Pre-training Loss Curve",
               os.path.join(outdir, "pretraining_loss_curve.png"))
    
    best_epoch = int(np.argmin(val_mse_losses)) if val_mse_losses else None
    metrics = {
        "training_losses": training_losses,
        "val_mse_losses": val_mse_losses,
        "val_mae_losses": val_mae_losses,
        "total_training_time_sec": total_training_time,
        "best_val_mse": float(best_val) if best_val != float('inf') else None,
        "best_epoch": best_epoch,
        "final_val_mse": float(val_mse_losses[-1]) if val_mse_losses else None,
        "min_val_mae": float(min(val_mae_losses)) if val_mae_losses else None,
    }
    return metrics

def analyze_model_performance(model, dataloader, device, metadata, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            mask_feat = batch["mask_feat"][0]
            targets = batch["targets"].to(device)
            
            masked_tokens = tokens.clone().masked_fill_(mask_feat.to(device), 0.0)
            preds = model(masked_tokens)
            
            means = torch.tensor(metadata['means'], device=device, dtype=torch.float32)
            stds = torch.tensor(metadata['stds'], device=device, dtype=torch.float32)
            
            all_preds.append((preds * stds + means).cpu().numpy())
            all_targets.append((targets * stds + means).cpu().numpy())
            all_masks.append(mask_feat.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_masks = np.concatenate(all_masks).astype(bool)
    
    vmag_preds = all_preds[:, :, -2][all_masks[:, :, -2]]
    vmag_actuals = all_targets[:, :, -2][all_masks[:, :, -2]]
    vang_preds = all_preds[:, :, -1][all_masks[:, :, -1]]
    vang_actuals = all_targets[:, :, -1][all_masks[:, :, -1]]
    
    # Scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Model Reconstruction Performance on Masked Buses", fontsize=18)
    
    ax1.scatter(vmag_actuals, vmag_preds, alpha=0.3, s=15, edgecolors='k', linewidth=0.5)
    ax1.plot([vmag_actuals.min(), vmag_actuals.max()], 
             [vmag_actuals.min(), vmag_actuals.max()], 'r--', lw=2, label="Ideal (y=x)")
    ax1.set_title("Voltage Magnitude (|V|)", fontsize=14)
    ax1.set_xlabel("Actual Value (p.u.)", fontsize=12)
    ax1.set_ylabel("Predicted Value (p.u.)", fontsize=12)
    ax1.grid(True); ax1.legend(); ax1.axis('equal')
    
    ax2.scatter(vang_actuals, vang_preds, alpha=0.3, s=15, edgecolors='k', linewidth=0.5)
    ax2.plot([vang_actuals.min(), vang_actuals.max()], 
             [vang_actuals.min(), vang_actuals.max()], 'r--', lw=2, label="Ideal (y=x)")
    ax2.set_title("Voltage Angle (θ)", fontsize=14)
    ax2.set_xlabel("Actual Value (degrees)", fontsize=12)
    ax2.set_ylabel("Predicted Value (degrees)", fontsize=12)
    ax2.grid(True); ax2.legend(); ax2.axis('equal')
    
    plt.savefig(os.path.join(save_dir, "reconstruction_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Error histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Distribution of Prediction Errors on Masked Buses", fontsize=18)
    
    ax1.hist(vmag_preds - vmag_actuals, bins=50, alpha=0.75, color='C0')
    ax1.set_title("Voltage Magnitude Error", fontsize=14)
    ax1.set_xlabel("Error (|V| pred - |V| actual)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12); ax1.grid(True)
    
    ax2.hist(vang_preds - vang_actuals, bins=50, alpha=0.75, color='C1')
    ax2.set_title("Voltage Angle Error", fontsize=14)
    ax2.set_xlabel("Error (θ pred - θ actual)", fontsize=12); ax2.grid(True)
    
    plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Analysis plots saved to {save_dir}")

def main():
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device.upper()}")
    
    # Auto-adjust batch size if requested
    actual_batch_size = args.batch_size
    if args.auto_adjust_batch:
        max_effective = 640
        effective = args.batch_size * args.num_views
        if effective > max_effective:
            actual_batch_size = max(1, max_effective // args.num_views)
            logging.info(f"Auto-adjusted batch size: {args.batch_size} -> {actual_batch_size} "
                        f"(for N={args.num_views})")
    
    data_path = args.data_path or "dataset_ieee30.npz"
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found: {data_path}")
        sys.exit(1)
    
    data_archive = np.load(data_path)
    token_dim = data_archive["X_tokens"].shape[2] + data_archive["Y_targets"].shape[2]
    num_buses = data_archive["X_tokens"].shape[1]
    logging.info(f"token_dim={token_dim}, num_buses={num_buses}, num_views={args.num_views}")
    
    loaders = get_dataloaders(data_path, batch_size_pre=actual_batch_size, 
                              mask_ratio=args.mask_ratio, num_views=args.num_views, 
                              seed=args.seed)
    
    metrics = pretrain_main(
        loaders, token_dim=token_dim, num_buses=num_buses, outdir=args.outdir,
        embed_dim=args.embed_dim, num_layers=args.num_layers, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
        device=device, memory_efficient=args.memory_efficient,
        accumulation_steps=args.accumulation_steps
    )
    
    logging.info("--- Starting Post-Training Analysis ---")
    best_model = MAEModel(token_dim, embed_dim=args.embed_dim, num_buses=num_buses, 
                         num_layers=args.num_layers).to(device)
    checkpoint = torch.load(os.path.join(args.outdir, "mae_best.pt"), map_location=device)
    best_model.load_state_dict(checkpoint["model_state"])
    
    analyze_model_performance(best_model, loaders["pretrain_val_loader"], device,
                             loaders["metadata"], os.path.join(args.outdir, "analysis_plots"))
    
    run_meta = {
        "seed": args.seed, "num_views": args.num_views, "token_dim": token_dim,
        "num_buses": num_buses, "device": device, "epochs": args.epochs,
        "embed_dim": args.embed_dim, "num_layers": args.num_layers,
        "batch_size": actual_batch_size, "mask_ratio": args.mask_ratio,
        "lr": args.lr, "weight_decay": args.weight_decay, "patience": args.patience,
        "memory_efficient": args.memory_efficient, 
        "accumulation_steps": args.accumulation_steps,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    metrics_all = {**run_meta, **metrics}
    
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)
    
    logging.info(f"Metrics saved to {os.path.join(args.outdir, 'metrics.json')}")

if __name__ == "__main__":
    main()