# check.py
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

# UPDATED: Configure logging to save to file and print to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training_log.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# UPDATED: plot_losses function to save high-quality figures
def plot_losses(train_mse, val_mse, val_mae, title="Loss Curve", save_path="loss_curve.png"):
    plt.figure(figsize=(12, 7))
    plt.plot(train_mse, label="Train masked MSE")
    plt.plot(val_mse, label="Val masked MSE")
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
        # We only want to reconstruct Vmag and Vang
        mask_feat[..., -2:] = masks.unsqueeze(-1).expand(-1, -1, -1, 2)
        return {"tokens": batch, "masks": masks, "mask_feat": mask_feat, "targets": batch.clone()}
    return collate_fn

def train_one_epoch_maskV_packed(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        tokens, mask_feat, targets = batch["tokens"].to(device), batch["mask_feat"].to(device), batch["targets"].to(device)
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

def validate_maskV_packed(model, dataloader, device):
    model.eval()
    total_mse, total_mae = 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            tokens, mask_feat, targets = batch["tokens"].to(device), batch["mask_feat"].to(device), batch["targets"].to(device)
            
            K, B, nb, td = mask_feat.shape
            
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


def get_dataloaders(npz_path, pretrain_frac=0.9, batch_size_pre=128, mask_ratio=0.75, num_views=2, seed=42):
    data = np.load(npz_path)
    # Combine X and Y to create the full "token" for self-supervision
    full_tokens = np.concatenate([data['X_tokens'], data['Y_targets']], axis=-1).astype(np.float32)

    # Normalize based on the training set only
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
    pretrain_loader = DataLoader(pretrain_train_ds, batch_size=batch_size_pre, shuffle=True, collate_fn=collate_fn)
    pretrain_val_loader = DataLoader(pretrain_val_ds, batch_size=batch_size_pre, shuffle=False, collate_fn=collate_fn)
    
    metadata = {"means": means, "stds": stds}
    return {"pretrain_loader": pretrain_loader, "pretrain_val_loader": pretrain_val_loader, "metadata": metadata}

class MAEModel(nn.Module):
    def __init__(self, token_dim, embed_dim, num_buses, num_layers=8, num_heads=8):
        super().__init__()
        self.token_embedding = nn.Linear(token_dim, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_buses, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(embed_dim, token_dim)

    def forward(self, x):
        x = self.token_embedding(x) + self.positional_embedding
        x = self.encoder(x)
        return self.decoder(x)

def pretrain_main(dataloaders, token_dim, num_buses, embed_dim=64, num_layers=4, epochs=200, lr=1e-4, device="cpu"):
    model = MAEModel(token_dim, embed_dim, num_buses, num_layers=num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val, patience_count = float('inf'), 0
    training_losses, val_mse_losses, val_mae_losses = [], [], []
    
    logging.info("--- Starting Pre-training ---")
    training_start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_one_epoch_maskV_packed(model, dataloaders['pretrain_loader'], optimizer, device)
        val_mse, val_mae = validate_maskV_packed(model, dataloaders['pretrain_val_loader'], device)

        training_losses.append(train_loss)
        val_mse_losses.append(val_mse)
        val_mae_losses.append(val_mae)
        
        # UPDATED: Use logging instead of print
        if(epoch%10==0):
            logging.info(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.6e} | Val MSE: {val_mse:.6e} | Val MAE: {val_mae:.6e}")

        scheduler.step(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            patience_count = 0
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, "mae_best.pt")
        else:
            patience_count += 1
            if patience_count >= 25: # Early stopping patience
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    total_training_time = time.time() - training_start_time
    logging.info(f"--- Pre-training Finished ---")
    logging.info(f"Total training time: {total_training_time // 60:.0f} minutes and {total_training_time % 60:.2f} seconds.")
    
    # UPDATED: Call the updated plotting function
    plot_losses(training_losses, val_mse_losses, val_mae_losses, title="Pre-training Loss Curve", save_path="pretraining_loss_curve.png")

# UPDATED: New analysis function to generate "cool plots"
def analyze_model_performance(model, dataloader, device, metadata, save_dir="analysis_plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            tokens, mask_feat, targets = batch["tokens"].to(device), batch["mask_feat"][0], batch["targets"].to(device)
            masked_tokens = tokens.clone().masked_fill_(mask_feat.to(tokens.device), 0.0)
            preds = model(masked_tokens)
            
            means = torch.tensor(metadata['means'], device=device, dtype=torch.float32)
            stds = torch.tensor(metadata['stds'], device=device, dtype=torch.float32)

            all_preds.append((preds * stds + means).cpu().numpy())
            all_targets.append((targets * stds + means).cpu().numpy())
            all_masks.append(mask_feat.cpu().numpy())

    all_preds, all_targets, all_masks = np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_masks).astype(bool)
    
    # Isolate V_mag (second to last) and V_ang (last) features
    vmag_preds, vmag_actuals = all_preds[:, :, -2][all_masks[:, :, -2]], all_targets[:, :, -2][all_masks[:, :, -2]]
    vang_preds, vang_actuals = all_preds[:, :, -1][all_masks[:, :, -1]], all_targets[:, :, -1][all_masks[:, :, -1]]

    # Plot 1: Scatter Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Model Reconstruction Performance on Masked Buses", fontsize=18)
    ax1.scatter(vmag_actuals, vmag_preds, alpha=0.3, s=15, edgecolors='k', linewidth=0.5)
    ax1.plot([vmag_actuals.min(), vmag_actuals.max()], [vmag_actuals.min(), vmag_actuals.max()], 'r--', lw=2, label="Ideal (y=x)")
    ax1.set_title("Voltage Magnitude (|V|)", fontsize=14); ax1.set_xlabel("Actual Value (p.u.)", fontsize=12); ax1.set_ylabel("Predicted Value (p.u.)", fontsize=12)
    ax1.grid(True); ax1.legend(); ax1.axis('equal')

    ax2.scatter(vang_actuals, vang_preds, alpha=0.3, s=15, edgecolors='k', linewidth=0.5)
    ax2.plot([vang_actuals.min(), vang_actuals.max()], [vang_actuals.min(), vang_actuals.max()], 'r--', lw=2, label="Ideal (y=x)")
    ax2.set_title("Voltage Angle (θ)", fontsize=14); ax2.set_xlabel("Actual Value (degrees)", fontsize=12); ax2.set_ylabel("Predicted Value (degrees)", fontsize=12)
    ax2.grid(True); ax2.legend(); ax2.axis('equal')
    plt.savefig(os.path.join(save_dir, "reconstruction_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Reconstruction scatter plot saved to {os.path.join(save_dir, 'reconstruction_scatter.png')}")

    # Plot 2: Error Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Distribution of Prediction Errors on Masked Buses", fontsize=18)
    ax1.hist(vmag_preds - vmag_actuals, bins=50, alpha=0.75, color='C0'); ax1.set_title("Voltage Magnitude Error", fontsize=14)
    ax1.set_xlabel("Error (|V| predicted - |V| actual)", fontsize=12); ax1.set_ylabel("Frequency", fontsize=12); ax1.grid(True)
    ax2.hist(vang_preds - vang_actuals, bins=50, alpha=0.75, color='C1'); ax2.set_title("Voltage Angle Error", fontsize=14)
    ax2.set_xlabel("Error (θ predicted - θ actual)", fontsize=12); ax2.grid(True)
    plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Error distribution plot saved to {os.path.join(save_dir, 'error_distribution.png')}")

if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = "dataset_ieee30.npz"
    
    # First, generate the data if it doesn't exist
    if not os.path.exists(DATA_PATH):
        logging.info(f"Dataset not found at {DATA_PATH}. Generating new data...")
        from generate_data import generate_data
        generate_data(num_samples=2000, output_path=DATA_PATH, seed=42)
        logging.info("Data generation complete.")

    data_archive = np.load(DATA_PATH)
    # The full token now includes X_tokens and Y_targets
    token_dim = data_archive['X_tokens'].shape[2] + data_archive['Y_targets'].shape[2]
    num_buses = data_archive['X_tokens'].shape[1]
    
    logging.info(f"Device: {device.upper()}")
    logging.info(f"Detected token_dim={token_dim}, num_buses={num_buses}")

    dataloaders = get_dataloaders(DATA_PATH)

    pretrain_main(
        dataloaders, token_dim=token_dim, num_buses=num_buses,
        epochs=1500, device=device, embed_dim=128, num_layers=6
    )

    # UPDATED: Post-training analysis block
    logging.info("--- Starting Post-Training Analysis ---")
    best_model = MAEModel(token_dim, embed_dim=128, num_buses=num_buses, num_layers=6).to(device)
    checkpoint = torch.load("mae_best.pt", map_location=device)
    best_model.load_state_dict(checkpoint['model_state'])
    
    analyze_model_performance(
        model=best_model,
        dataloader=dataloaders['pretrain_val_loader'],
        device=device,
        metadata=dataloaders['metadata']
    )
    logging.info("--- Analysis Complete ---")