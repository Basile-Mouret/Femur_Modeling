#!/usr/bin/env python3
"""
Tangent PCA Complete Pipeline Demo

This script demonstrates the full LDDMM-based Tangent PCA pipeline:
1. Load femur data
2. Build atlas (Fréchet mean)
3. Fit Tangent PCA
4. Visualize results

Run this script to generate a complete Tangent PCA model and explore the shape space.

Usage:
    python tangent_pca_demo.py

Author: Femur Modeling Project
Date: January 2026
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add path for imports (project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lddmm import (
    FemurDataLoader,
    LDDMMAtlasBuilder,
    TangentPCA
)


def main():
    """Run the complete Tangent PCA pipeline."""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "training"
    model_dir = Path(__file__).parent / "model" / "tangent_pca"
    
    print("=" * 70)
    print(" LDDMM-based Tangent PCA Pipeline Demo")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[1/4] Loading femur data...")
    
    loader = FemurDataLoader(str(data_dir))
    shapes, filenames = loader.load_all()
    
    print(f"      Loaded {len(shapes)} shapes")
    print(f"      Shape dimensions: {shapes[0].shape}")
    
    # =========================================================================
    # Step 2: Build Atlas
    # =========================================================================
    print("\n[2/4] Building atlas (Fréchet mean)...")
    
    atlas_builder = LDDMMAtlasBuilder(
        max_outer_iterations=3,  # Quick demo
        convergence_tol=1e-3,
        verbose=True
    )
    
    result = atlas_builder.build(shapes)
    atlas = result['atlas']
    momenta = result['momenta']
    
    print(f"      Atlas shape: {atlas.shape}")
    print(f"      Number of momenta: {len(momenta)}")
    
    # Save atlas
    atlas_dir = model_dir / "atlas"
    atlas_builder.save(str(atlas_dir))
    print(f"      Atlas saved to: {atlas_dir}")
    
    # =========================================================================
    # Step 3: Fit Tangent PCA
    # =========================================================================
    print("\n[3/4] Fitting Tangent PCA...")
    
    pca = TangentPCA(n_components=10)
    pca.fit(atlas, momenta)
    
    print(f"      Components: {pca.n_components}")
    print(f"      Samples: {pca.n_samples}")
    
    # Print variance explained
    print("\n      Variance explained by each component:")
    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio[:10]):
        cumulative += var
        print(f"        PC{i+1}: {var*100:6.2f}%  (cumulative: {cumulative*100:6.2f}%)")
    
    # Save model
    pca.save(str(model_dir))
    print(f"\n      Model saved to: {model_dir}")
    
    # =========================================================================
    # Step 4: Generate Sample Shapes
    # =========================================================================
    print("\n[4/4] Generating sample shapes...")
    
    # Generate shapes along PC1
    t_values = np.array([-2, -1, 0, 1, 2])
    pc1_shapes = pca.synthesize_along_mode(mode=0, t_values=t_values)
    
    print(f"      Generated {len(pc1_shapes)} shapes along PC1")
    print(f"      Shape dimensions: {pc1_shapes[0].shape}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Pipeline Complete!")
    print("=" * 70)
    print(f"""
Summary:
  - Data: {len(shapes)} femur shapes ({shapes[0].shape[0]} vertices each)
  - Atlas: Fréchet mean computed
  - Tangent PCA: {pca.n_components} components
  - Top variance: PC1={pca.explained_variance_ratio[0]*100:.1f}%
  
Model saved to: {model_dir}

To explore interactively, run:
  python tangent_pca_explorer.py --model {model_dir} --template {data_dir}/L_Femur_11.obj

To visualize modes:
  python tangent_pca_visualizer.py --model {model_dir} --template {data_dir}/L_Femur_11.obj --variance
  python tangent_pca_visualizer.py --model {model_dir} --template {data_dir}/L_Femur_11.obj --mode 0
""")


if __name__ == '__main__':
    main()
