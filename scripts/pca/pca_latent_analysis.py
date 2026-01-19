#!/usr/bin/env python3
"""
PCA Analysis of Latent Space

Performs PCA on the latent space projections and visualizes
the explained variance per principal component.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths
script_dir = Path(__file__).parent.absolute()
data_file = script_dir / "latent_projections.npz"


def load_projections():
    """Load latent projections from saved file."""
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please run 'project_training_femurs.py' first.")
        sys.exit(1)
    
    data = np.load(data_file, allow_pickle=True)
    latents = data['latents']
    femur_names = data['femur_names']
    return latents, femur_names


def main():
    # Load data
    print("Loading latent projections...")
    latents, femur_names = load_projections()
    n_samples, n_dims = latents.shape
    print(f"Loaded {n_samples} femurs with {n_dims}-dimensional latent space")
    print()
    
    # Standardize the data (center and scale)
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)
    
    # Perform PCA
    pca = PCA()
    pca.fit(latents_scaled)
    
    # Get results
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    eigenvalues = pca.explained_variance_
    
    # Print results
    print("=" * 60)
    print("PCA RESULTS")
    print("=" * 60)
    print()
    print(f"{'PC':<6} {'Eigenvalue':>12} {'Var. Explained':>15} {'Cumulative':>12}")
    print("-" * 50)
    for i in range(n_dims):
        print(f"PC{i:<4} {eigenvalues[i]:>12.6f} {explained_var[i]:>14.2%} {cumulative_var[i]:>11.2%}")
    print("-" * 50)
    print()
    
    # Find number of components for different thresholds
    for thresh in [0.90, 0.95, 0.99, 0.999]:
        n_comp = np.argmax(cumulative_var >= thresh) + 1
        print(f"Components for {thresh:.1%} variance: {n_comp}")
    print()
    
    # Create figure with multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PCA Analysis of Latent Space', fontsize=14, fontweight='bold')
    
    # Plot 1: Scree plot (eigenvalues)
    ax1 = axes[0, 0]
    ax1.bar(range(1, n_dims + 1), eigenvalues, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Eigenvalue', fontsize=11)
    ax1.set_title('Scree Plot (Eigenvalues)', fontsize=12)
    ax1.set_xticks(range(1, n_dims + 1))
    ax1.set_xticklabels([f'PC{i}' for i in range(n_dims)], rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Explained variance ratio
    ax2 = axes[0, 1]
    bars = ax2.bar(range(1, n_dims + 1), explained_var * 100, color='coral', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Principal Component', fontsize=11)
    ax2.set_ylabel('Explained Variance (%)', fontsize=11)
    ax2.set_title('Explained Variance per Component', fontsize=12)
    ax2.set_xticks(range(1, n_dims + 1))
    ax2.set_xticklabels([f'PC{i}' for i in range(n_dims)], rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, var in zip(bars, explained_var):
        height = bar.get_height()
        ax2.annotate(f'{var:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Cumulative variance
    ax3 = axes[1, 0]
    ax3.plot(range(1, n_dims + 1), cumulative_var * 100, 'o-', color='green', 
             linewidth=2, markersize=8, label='Cumulative')
    ax3.bar(range(1, n_dims + 1), explained_var * 100, color='lightgreen', 
            alpha=0.5, edgecolor='green', label='Individual')
    
    # Add threshold lines
    for thresh, color, style in [(90, 'orange', '--'), (95, 'red', '--'), (99, 'purple', ':')]:
        ax3.axhline(y=thresh, color=color, linestyle=style, alpha=0.7, label=f'{thresh}% threshold')
    
    ax3.set_xlabel('Number of Components', fontsize=11)
    ax3.set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
    ax3.set_title('Cumulative Explained Variance', fontsize=12)
    ax3.set_xticks(range(1, n_dims + 1))
    ax3.set_xticklabels([f'{i}' for i in range(1, n_dims + 1)])
    ax3.set_ylim(0, 105)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(alpha=0.3)
    
    # Plot 4: Component loadings heatmap
    ax4 = axes[1, 1]
    loadings = pca.components_.T  # Shape: (n_features, n_components)
    im = ax4.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xlabel('Principal Component', fontsize=11)
    ax4.set_ylabel('Original Latent Dimension', fontsize=11)
    ax4.set_title('PCA Loadings (Contribution of each z to each PC)', fontsize=12)
    ax4.set_xticks(range(n_dims))
    ax4.set_xticklabels([f'PC{i}' for i in range(n_dims)])
    ax4.set_yticks(range(n_dims))
    ax4.set_yticklabels([f'z{i}' for i in range(n_dims)])
    plt.colorbar(im, ax=ax4, shrink=0.8, label='Loading')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = script_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "pca_variance_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Figure saved: {output_path}")
    
    # Save only the cumulative variance plot (bottom left)
    fig_single, ax_single = plt.subplots(figsize=(8, 6))
    ax_single.plot(range(1, n_dims + 1), cumulative_var * 100, 'o-', color='green', 
             linewidth=2, markersize=8, label='Cumulative')
    ax_single.bar(range(1, n_dims + 1), explained_var * 100, color='lightgreen', 
            alpha=0.5, edgecolor='green', label='Individual')
    
    # Add threshold lines
    for thresh, color, style in [(90, 'orange', '--'), (95, 'red', '--'), (99, 'purple', ':')]:
        ax_single.axhline(y=thresh, color=color, linestyle=style, alpha=0.7, label=f'{thresh}% threshold')
    
    ax_single.set_xlabel('Number of Components', fontsize=11)
    ax_single.set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
    ax_single.set_title('Cumulative Explained Variance', fontsize=12)
    ax_single.set_xticks(range(1, n_dims + 1))
    ax_single.set_xticklabels([f'{i}' for i in range(1, n_dims + 1)])
    ax_single.set_ylim(0, 105)
    ax_single.legend(loc='lower right', fontsize=9)
    ax_single.grid(alpha=0.3)
    
    fig_single.tight_layout()
    output_path_single = output_dir / "pca_cumulative_variance.png"
    fig_single.savefig(output_path_single, dpi=150, bbox_inches='tight')
    print(f"✅ Cumulative variance plot saved: {output_path_single}")
    
    # Show
    plt.show()
    
    return pca, explained_var, cumulative_var


if __name__ == "__main__":
    pca, explained_var, cumulative_var = main()
