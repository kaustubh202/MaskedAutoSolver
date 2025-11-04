#!/usr/bin/env python3
# analyze_ablations.py
"""
Comprehensive analysis script for N×M ablation study.
Aggregates metrics, creates publication-quality plots, and generates LaTeX tables.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, List, Tuple

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
})

def load_all_metrics(results_dir: Path) -> pd.DataFrame:
    """Load all metrics.json files from the results directory."""
    records = []
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Parse directory name: n_X_m_Y
        parts = exp_dir.name.split('_')
        if len(parts) != 4 or parts[0] != 'n' or parts[2] != 'm':
            continue
        
        try:
            n = int(parts[1])
            m = int(parts[3])
        except ValueError:
            continue
        
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        try:
            with metrics_file.open('r') as f:
                data = json.load(f)
            
            # Extract key metrics
            record = {
                'N': n,
                'M': m,
                'best_val_mse': data.get('best_val_mse'),
                'final_val_mse': data.get('final_val_mse'),
                'min_val_mae': data.get('min_val_mae'),
                'best_epoch': data.get('best_epoch'),
                'training_time_sec': data.get('total_training_time_sec'),
                'converged': data.get('best_val_mse') is not None,
            }
            
            # Extract final losses if available
            if 'val_mse_losses' in data and data['val_mse_losses']:
                record['final_val_mse'] = data['val_mse_losses'][-1]
            if 'val_mae_losses' in data and data['val_mae_losses']:
                record['final_val_mae'] = data['val_mae_losses'][-1]
            
            records.append(record)
        
        except Exception as e:
            print(f"Warning: Could not load {metrics_file}: {e}")
            continue
    
    df = pd.DataFrame(records)
    return df.sort_values(['N', 'M']).reset_index(drop=True)

def plot_mse_heatmap(df: pd.DataFrame, save_path: Path, metric='best_val_mse'):
    """Create a heatmap of MSE across N and M."""
    pivot = df.pivot(index='N', columns='M', values=metric)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use log scale for better visualization
    im = ax.imshow(np.log10(pivot.values + 1e-10), cmap='viridis', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel('M (Dataset Size)', fontsize=14)
    ax.set_ylabel('N (Number of Views)', fontsize=14)
    ax.set_title(f'{metric.replace("_", " ").title()} (Log Scale)', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(MSE)', fontsize=12)
    
    # Annotate cells with actual values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.2e}',
                             ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {save_path}")

def plot_n_ablation(df: pd.DataFrame, save_path: Path):
    """Plot MSE vs N for different M values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique M values
    m_values = sorted(df['M'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(m_values)))
    
    for m, color in zip(m_values, colors):
        subset = df[df['M'] == m].sort_values('N')
        
        # Plot best MSE
        ax1.plot(subset['N'], subset['best_val_mse'], 
                marker='o', label=f'M={m}', color=color, linewidth=2)
        
        # Plot final MSE
        if 'final_val_mse' in subset.columns:
            ax2.plot(subset['N'], subset['final_val_mse'], 
                    marker='s', label=f'M={m}', color=color, linewidth=2)
    
    ax1.set_xlabel('N (Number of Views)', fontsize=12)
    ax1.set_ylabel('Best Validation MSE', fontsize=12)
    ax1.set_title('Best MSE vs Number of Views', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('N (Number of Views)', fontsize=12)
    ax2.set_ylabel('Final Validation MSE', fontsize=12)
    ax2.set_title('Final MSE vs Number of Views', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved N ablation plot to {save_path}")

def plot_m_ablation(df: pd.DataFrame, save_path: Path):
    """Plot MSE vs M for different N values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique N values
    n_values = sorted(df['N'].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(n_values)))
    
    for n, color in zip(n_values, colors):
        subset = df[df['N'] == n].sort_values('M')
        
        # Plot best MSE
        ax1.plot(subset['M'], subset['best_val_mse'], 
                marker='o', label=f'N={n}', color=color, linewidth=2)
        
        # Plot final MAE
        if 'min_val_mae' in subset.columns:
            ax2.plot(subset['M'], subset['min_val_mae'], 
                    marker='s', label=f'N={n}', color=color, linewidth=2)
    
    ax1.set_xlabel('M (Dataset Size)', fontsize=12)
    ax1.set_ylabel('Best Validation MSE', fontsize=12)
    ax1.set_title('Best MSE vs Dataset Size', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('M (Dataset Size)', fontsize=12)
    ax2.set_ylabel('Min Validation MAE', fontsize=12)
    ax2.set_title('MAE vs Dataset Size', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved M ablation plot to {save_path}")

def plot_training_efficiency(df: pd.DataFrame, save_path: Path):
    """Plot training time and convergence speed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training time vs N for different M
    m_values = sorted(df['M'].unique())[:5]  # Top 5 M values
    colors = plt.cm.viridis(np.linspace(0, 1, len(m_values)))
    
    for m, color in zip(m_values, colors):
        subset = df[df['M'] == m].sort_values('N')
        if 'training_time_sec' in subset.columns:
            ax1.plot(subset['N'], subset['training_time_sec'] / 60, 
                    marker='o', label=f'M={m}', color=color, linewidth=2)
    
    ax1.set_xlabel('N (Number of Views)', fontsize=12)
    ax1.set_ylabel('Training Time (minutes)', fontsize=12)
    ax1.set_title('Training Time vs Number of Views', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Convergence speed (epochs to best)
    for m, color in zip(m_values, colors):
        subset = df[df['M'] == m].sort_values('N')
        if 'best_epoch' in subset.columns:
            ax2.plot(subset['N'], subset['best_epoch'], 
                    marker='s', label=f'M={m}', color=color, linewidth=2)
    
    ax2.set_xlabel('N (Number of Views)', fontsize=12)
    ax2.set_ylabel('Epochs to Best Model', fontsize=12)
    ax2.set_title('Convergence Speed vs Number of Views', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved efficiency plot to {save_path}")

def plot_data_efficiency(df: pd.DataFrame, save_path: Path):
    """
    Plot effective samples (M * N) vs performance.
    This shows if more views can compensate for fewer solved scenarios.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    df['effective_samples'] = df['M'] * df['N']
    
    # Color by N
    n_values = sorted(df['N'].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(n_values)))
    
    for n, color in zip(n_values, colors):
        subset = df[df['N'] == n].sort_values('effective_samples')
        ax.scatter(subset['effective_samples'], subset['best_val_mse'],
                  s=100, label=f'N={n}', color=color, alpha=0.7, edgecolors='k')
        ax.plot(subset['effective_samples'], subset['best_val_mse'],
               color=color, alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Effective Samples (M × N)', fontsize=12)
    ax.set_ylabel('Best Validation MSE', fontsize=12)
    ax.set_title('Data Efficiency: Does N compensate for small M?', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved data efficiency plot to {save_path}")

def generate_latex_table(df: pd.DataFrame, save_path: Path):
    """Generate LaTeX table for paper."""
    # Select key M and N values for the table
    key_m = [10000, 5000, 2000, 1000, 500, 200, 100]
    key_n = [1, 2, 3, 5, 10, 15, 20]
    
    # Filter dataframe
    table_df = df[df['M'].isin(key_m) & df['N'].isin(key_n)]
    
    # Create pivot table
    pivot = table_df.pivot(index='N', columns='M', values='best_val_mse')
    
    # Generate LaTeX
    latex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Best Validation MSE for different combinations of dataset size (M) and number of views (N). Lower is better.}",
        "\\label{tab:ablation_mse}",
        "\\begin{tabular}{c" + "c" * len(pivot.columns) + "}",
        "\\toprule",
    ]
    
    # Header
    header = "N \\\\ M & " + " & ".join([f"{m}" for m in pivot.columns]) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Data rows
    for n in pivot.index:
        row = [f"{n}"]
        for m in pivot.columns:
            val = pivot.loc[n, m]
            if pd.isna(val):
                row.append("--")
            else:
                row.append(f"{val:.2e}")
        latex_lines.append(" & ".join(row) + " \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    with save_path.open('w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"Saved LaTeX table to {save_path}")

def generate_summary_stats(df: pd.DataFrame, save_path: Path):
    """Generate summary statistics."""
    summary = {
        "Total experiments": len(df),
        "Converged experiments": int(df['converged'].sum()),
        "N values tested": sorted(df['N'].unique().tolist()),
        "M values tested": sorted(df['M'].unique().tolist()),
        "Best overall MSE": float(df['best_val_mse'].min()),
        "Best (N, M)": tuple(df.loc[df['best_val_mse'].idxmin(), ['N', 'M']].astype(int).tolist()),
        "Median MSE": float(df['best_val_mse'].median()),
        "Mean training time (min)": float(df['training_time_sec'].mean() / 60) if 'training_time_sec' in df else None,
    }
    
    # Find trends
    if len(df['N'].unique()) > 1:
        # Average MSE improvement from N=1 to max N for each M
        improvements = []
        for m in df['M'].unique():
            subset = df[df['M'] == m].sort_values('N')
            if len(subset) > 1:
                n1_mse = subset[subset['N'] == subset['N'].min()]['best_val_mse'].values[0]
                nmax_mse = subset[subset['N'] == subset['N'].max()]['best_val_mse'].values[0]
                improvement = (n1_mse - nmax_mse) / n1_mse * 100
                improvements.append(improvement)
        if improvements:
            summary["Avg MSE improvement (%) from min to max N"] = float(np.mean(improvements))
    
    with save_path.open('w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary Statistics:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="analysis",
                       help="Directory to save analysis outputs")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading metrics from all experiments...")
    df = load_all_metrics(results_dir)
    
    if len(df) == 0:
        print("ERROR: No valid experiments found!")
        return
    
    print(f"Loaded {len(df)} experiments")
    print(f"N values: {sorted(df['N'].unique())}")
    print(f"M values: {sorted(df['M'].unique())}")
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_mse_heatmap(df, output_dir / "mse_heatmap.png")
    plot_n_ablation(df, output_dir / "n_ablation.png")
    plot_m_ablation(df, output_dir / "m_ablation.png")
    plot_training_efficiency(df, output_dir / "training_efficiency.png")
    plot_data_efficiency(df, output_dir / "data_efficiency.png")
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(df, output_dir / "ablation_table.tex")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    generate_summary_stats(df, output_dir / "summary_stats.json")
    
    # Save processed dataframe
    df.to_csv(output_dir / "all_metrics.csv", index=False)
    print(f"\nSaved all metrics to {output_dir / 'all_metrics.csv'}")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()