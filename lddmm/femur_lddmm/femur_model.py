#!/usr/bin/env python3
"""
Build and save a Tangent PCA model for femur shape analysis.

This script:
1. Loads femur meshes with established point correspondence
2. Builds an atlas (population mean shape)
3. Computes Tangent PCA for shape variation analysis
4. Saves the complete model to disk

Warning : This process can be computationally intensive. It is designed to run on GPU-enabled systems for efficiency, using PyKeops via scikit-shapes.
This library needs full CUDA toolkit installed and a compatible GPU.

Usage (from project root):
    python -m lddmm.femur_lddmm.femur_model                    # Use defaults
    python -m lddmm.femur_lddmm.femur_model --n-components 15  # Keep 15 PCA components
    python -m lddmm.femur_lddmm.femur_model --config fast      # Use fast preset
    python -m lddmm.femur_lddmm.femur_model --help             # Show all options

Output:
    models/lddmm_pca/
    ├── atlas.npy                        # Mean shape (N, 3)
    ├── momenta.npy                      # Momenta to each training shape (K, N, 3)
    ├── atlas_metadata.json              # Atlas metadata
    ├── tangent_pca_atlas.npy            # Duplicate for TangentPCA
    ├── tangent_pca_components.npy       # Principal components (n_components, N, 3)
    ├── tangent_pca_eigenvalues.npy      # Variance per component
    ├── tangent_pca_explained_variance.npy
    ├── tangent_pca_mean_momentum.npy
    └── tangent_pca_metadata.json
"""

import argparse
import sys
from pathlib import Path

from .. import (
    FemurDataLoader,
    AtlasBuilder,
    TangentPCA,
    LDDMMConfig,
    verify_correspondence,
    compute_bounding_box,
)

# Project root for resolving data/output paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Tangent PCA model for femur shapes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                                        # Use all data (training + validation)
    %(prog)s --n-components 20                      # Keep 20 PCA components
    %(prog)s --config high_precision                # More accurate LDDMM
        """,
    )

    parser.add_argument(
        "--data-dirs",
        type=str,
        nargs="+",
        default=["data/training", "data/validation"],
        help="Directories containing OBJ files (default: data/training data/validation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/lddmm_pca",
        help="Output directory for model (default: models/lddmm_pca)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of PCA components to keep (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["for_femurs", "high_precision", "fast"],
        default="for_femurs",
        help="LDDMM configuration preset (default: for_femurs)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override kernel scale parameter (mm)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    return parser.parse_args()


def get_config(args: argparse.Namespace) -> LDDMMConfig:
    """Get LDDMM configuration from args."""
    if args.config == "for_femurs":
        config = LDDMMConfig.for_femurs()
    elif args.config == "high_precision":
        config = LDDMMConfig.high_precision()
    elif args.config == "fast":
        config = LDDMMConfig.fast()
    else:
        config = LDDMMConfig()

    # Override scale if specified
    if args.scale is not None:
        config.scale = args.scale

    return config


def load_from_directories(data_dirs: list, verbose: bool = True) -> tuple:
    """
    Load shapes from multiple directories.

    Args:
        data_dirs: List of directory paths containing OBJ files.
        verbose: Print progress information.

    Returns:
        Tuple of (shapes, filenames) combined from all directories.
    """
    all_shapes = []
    all_filenames = []

    for data_dir in data_dirs:
        if not data_dir.exists():
            if verbose:
                print(f"  Skipping {data_dir} (not found)")
            continue

        try:
            loader = FemurDataLoader(str(data_dir))
            shapes, filenames = loader.load_all()
            all_shapes.extend(shapes)
            all_filenames.extend(filenames)
            if verbose:
                print(f"  {data_dir.name}: {len(shapes)} shapes")
        except ValueError as e:
            if verbose:
                print(f"  Skipping {data_dir}: {e}")

    return all_shapes, all_filenames


def main() -> int:
    """Main entry point."""
    args = parse_args()
    verbose = not args.quiet

    # Resolve paths relative to project root
    data_dirs = [PROJECT_ROOT / d for d in args.data_dirs]
    output_dir = PROJECT_ROOT / args.output_dir

    if verbose:
        print("=" * 60)
        print("Femur Tangent PCA Model Builder")
        print("=" * 60)
        print(f"\nData directories: {[str(d) for d in data_dirs]}")
        print(f"Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[1/4] Loading femur data...")

    shapes, filenames = load_from_directories(data_dirs, verbose)

    if len(shapes) == 0:
        print("Error: No shapes loaded from any directory")
        return 1

    if verbose:
        print(f"  Total: {len(shapes)} shapes")
        print(f"  Vertices per shape: {shapes[0].shape[0]}")

    # Verify correspondence
    if not verify_correspondence(shapes, verbose=verbose):
        print("Error: Shapes do not have point correspondence!")
        return 1

    # Compute bounding box for diagnostics
    bbox = compute_bounding_box(shapes)
    diagonal = float((bbox["size"] ** 2).sum() ** 0.5)

    if verbose:
        print(f"  Bounding box diagonal: {diagonal:.1f} mm")

    # -------------------------------------------------------------------------
    # 2. Configure LDDMM
    # -------------------------------------------------------------------------
    config = get_config(args)

    if verbose:
        print(f"\n[2/4] LDDMM Configuration ({args.config}):")
        print(f"  Kernel: {config.kernel}")
        print(f"  Scale: {config.scale} mm")
        print(f"  N steps: {config.n_steps}")
        print(f"  Regularization: {config.regularization_weight}")
        print(f"  Device: {config.device}")

    # -------------------------------------------------------------------------
    # 3. Build atlas
    # -------------------------------------------------------------------------
    if verbose:
        print("\n[3/4] Building atlas (average shape w.r.t geodesic distance)...")

    builder = AtlasBuilder(
        config=config,
        verbose=verbose,
    )
    atlas_result = builder.build(shapes)

    if verbose:
        print(f"  Atlas shape: {atlas_result.atlas.shape}")
        print(f"  Momenta shape: {atlas_result.momenta.shape}")

    # -------------------------------------------------------------------------
    # 4. Compute Tangent PCA
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n[4/4] Computing Tangent PCA...")

    pca = TangentPCA(n_components=args.n_components)
    pca.fit(atlas_result.atlas, atlas_result.momenta)

    # -------------------------------------------------------------------------
    # 5. Save model
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\nSaving model to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save atlas
    builder.save(str(output_dir))

    # Save PCA
    pca.save(str(output_dir))

    # Save additional metadata
    import json

    model_info = {
        "data_dirs": [str(d) for d in data_dirs],
        "n_training_shapes": len(shapes),
        "n_vertices": shapes[0].shape[0],
        "filenames": filenames,
        "config": {
            "preset": args.config,
            "kernel": config.kernel,
            "scale": config.scale,
            "n_steps": config.n_steps,
            "regularization_weight": config.regularization_weight,
        },
        "atlas_method": "geodesic",
        "n_pca_components": pca.n_components,
        "explained_variance_cumsum": [
            float(v) for v in pca.explained_variance_ratio.cumsum()
        ],
    }

    with open(output_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("Model Summary")
        print("=" * 60)
        print(f"Training shapes:     {len(shapes)}")
        print(f"Vertices per shape:  {shapes[0].shape[0]}")
        print(f"PCA components:      {pca.n_components}")
        print(f"Variance explained:")
        for i in range(min(5, pca.n_components)):
            cumvar = pca.explained_variance_ratio[:i + 1].sum() * 100
            print(f"  Mode {i}: {pca.explained_variance_ratio[i]*100:.1f}% (cumulative: {cumvar:.1f}%)")
        total_var = pca.explained_variance_ratio.sum() * 100
        print(f"  Total:  {total_var:.1f}%")
        print(f"\nModel saved to: {output_dir}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
