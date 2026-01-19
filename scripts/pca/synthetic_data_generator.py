#!/usr/bin/env python3
"""
Synthetic Data Generator for PCA-based Femur Models

This module generates synthetic femur shapes from a trained PCA model
and exports them in OBJ format compatible with the C++ framework.

Generation Strategies:
- Random sampling: Normal distribution in latent space
- Extreme modes: Shapes at ±k sigma along each principal component
- Latin Hypercube Sampling: Space-filling design for comprehensive coverage
- Grid sampling: Regular grid in the first few PC dimensions

Usage:
    python synthetic_data_generator.py --model model/pca_femur_model.bin \\
        --template ../data/training/L_Femur_11_DECIM.obj.FINAL.obj \\
        --output ../data/synthetic --count 100 --strategy random

Author: Femur Modeling Project
Date: 2026
"""

import os
import sys
import argparse
import struct
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from pathlib import Path

# Try to import scipy for Latin Hypercube Sampling
try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Constants
# =============================================================================

# Standardization factors from C++ Femur::getCoordsVect
STANDARDIZATION_FACTORS = {
    'x': 152.2,
    'y': 20.4,
    'z': 16.2
}


# =============================================================================
# PCA Model Loading
# =============================================================================

@dataclass
class PCAModel:
    """PCA model loaded from C++ binary file."""
    mean: np.ndarray           # Mean shape (D,)
    components: np.ndarray     # Principal components (D, K)
    variances: np.ndarray      # Variance per component (K,)
    n_dimensions: int          # D
    n_samples: int             # N (training samples)
    n_components: int          # K
    total_variance: float


def load_pca_model(filepath: str) -> PCAModel:
    """
    Load a PCA model from binary file (C++ PCA class format).
    
    Args:
        filepath: Path to the .bin file
        
    Returns:
        PCAModel object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PCA model not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        magic = f.read(4).decode('ascii')
        if magic != 'PCA1':
            raise ValueError(f"Invalid PCA file format. Expected 'PCA1', got '{magic}'")
        
        n_dimensions = struct.unpack('Q', f.read(8))[0]
        n_samples = struct.unpack('Q', f.read(8))[0]
        n_components = struct.unpack('Q', f.read(8))[0]
        total_variance = struct.unpack('d', f.read(8))[0]
        
        mean = np.array(struct.unpack(f'{n_dimensions}d', f.read(n_dimensions * 8)))
        variances = np.array(struct.unpack(f'{n_components}d', f.read(n_components * 8)))
        
        components_flat = np.array(struct.unpack(
            f'{n_dimensions * n_components}d',
            f.read(n_dimensions * n_components * 8)
        ))
        components = components_flat.reshape((n_components, n_dimensions)).T
    
    return PCAModel(
        mean=mean,
        components=components,
        variances=variances,
        n_dimensions=n_dimensions,
        n_samples=n_samples,
        n_components=n_components,
        total_variance=total_variance
    )


# =============================================================================
# Shape Generation
# =============================================================================

def generate_shape(model: PCAModel, weights: np.ndarray) -> np.ndarray:
    """
    Generate a shape from PCA weights.
    
    Shape = mean + sum_k(weight_k * sqrt(variance_k) * component_k)
    
    Args:
        model: PCA model
        weights: Weight for each component (in units of std deviation)
        
    Returns:
        Generated shape vector (D,) in standardized coordinates
    """
    n_weights = min(len(weights), model.n_components)
    shape = model.mean.copy()
    
    for k in range(n_weights):
        shape += weights[k] * np.sqrt(model.variances[k]) * model.components[:, k]
    
    return shape


def shape_to_points(shape: np.ndarray, destandardize: bool = True) -> np.ndarray:
    """
    Convert flattened shape vector to (N, 3) points.
    
    The C++ code stores coordinates as [all_X, all_Y, all_Z] (stacked).
    
    Args:
        shape: Flattened shape vector (D,)
        destandardize: Whether to multiply by standardization factors
        
    Returns:
        Points array (N, 3)
    """
    n_vertices = len(shape) // 3
    
    x_coords = shape[0:n_vertices]
    y_coords = shape[n_vertices:2*n_vertices]
    z_coords = shape[2*n_vertices:3*n_vertices]
    
    if destandardize:
        x_coords = x_coords * STANDARDIZATION_FACTORS['x']
        y_coords = y_coords * STANDARDIZATION_FACTORS['y']
        z_coords = z_coords * STANDARDIZATION_FACTORS['z']
    
    return np.column_stack([x_coords, y_coords, z_coords])


# =============================================================================
# OBJ File I/O
# =============================================================================

def load_template_obj(filepath: str) -> Tuple[np.ndarray, List[Tuple[int, int, int]], List[np.ndarray]]:
    """
    Load template OBJ file to get face connectivity and normals.
    
    Args:
        filepath: Path to OBJ file
        
    Returns:
        Tuple of (vertices, faces, normals)
        - vertices: (N, 3) array
        - faces: List of (v1, v2, v3) tuples (1-indexed)
        - normals: (N, 3) array or empty list
    """
    vertices = []
    normals = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn':
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Handle face format: "f v1 v2 v3" or "f v1//vn1 v2//vn2 v3//vn3"
                face_indices = []
                for p in parts[1:4]:
                    idx = int(p.split('/')[0])
                    face_indices.append(idx)
                faces.append(tuple(face_indices))
    
    return np.array(vertices), faces, np.array(normals) if normals else []


def save_shape_to_obj(
    filepath: str,
    points: np.ndarray,
    faces: List[Tuple[int, int, int]],
    normals: Optional[np.ndarray] = None,
    header: str = ""
) -> None:
    """
    Save a shape to OBJ file format compatible with C++ Femur class.
    
    Args:
        filepath: Output file path
        points: Vertex coordinates (N, 3)
        faces: Face indices (1-indexed as in OBJ format)
        normals: Optional normal vectors (N, 3)
        header: Optional header comment
    """
    with open(filepath, 'w') as f:
        # Header comment
        if header:
            f.write(f"# {header}\n")
        f.write(f"# {len(points)} vertice(s)\n")
        
        # Write vertices
        for v in points:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write normals if provided
        if normals is not None and len(normals) > 0:
            for n in normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        
        # Write faces
        for face in faces:
            if normals is not None and len(normals) > 0:
                # Include normal indices (same as vertex indices for per-vertex normals)
                f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")
            else:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")


# =============================================================================
# Sampling Strategies
# =============================================================================

class SyntheticGenerator:
    """
    Generator for synthetic femur shapes from a PCA model.
    
    Supports multiple sampling strategies:
    - random: Sample weights from multivariate normal
    - extreme: Generate shapes at ±k sigma along each PC
    - lhs: Latin Hypercube Sampling for space-filling design
    - grid: Regular grid in first n dimensions
    """
    
    def __init__(
        self,
        model: PCAModel,
        template_path: str,
        n_components: Optional[int] = None,
        sigma_range: float = 3.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the generator.
        
        Args:
            model: Loaded PCA model
            template_path: Path to template OBJ for face connectivity
            n_components: Number of components to use (default: all)
            sigma_range: Range for sampling (default: ±3 sigma)
            seed: Random seed for reproducibility
        """
        self.model = model
        self.n_components = n_components or model.n_components
        self.sigma_range = sigma_range
        self.rng = np.random.default_rng(seed)
        
        # Load template for face connectivity
        self.template_vertices, self.template_faces, self.template_normals = \
            load_template_obj(template_path)
        
        print(f"[Generator] Initialized with {self.n_components} components, "
              f"σ_range=±{sigma_range}")
        print(f"[Generator] Template: {len(self.template_vertices)} vertices, "
              f"{len(self.template_faces)} faces")
    
    def generate_random(self, count: int) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate shapes by random sampling from normal distribution.
        
        Weights are sampled from N(0, 1) and clipped to sigma_range.
        
        Args:
            count: Number of shapes to generate
            
        Returns:
            List of (weights, metadata) tuples
        """
        samples = []
        for i in range(count):
            # Sample from standard normal, clip to valid range
            weights = self.rng.standard_normal(self.n_components)
            weights = np.clip(weights, -self.sigma_range, self.sigma_range)
            
            metadata = {
                'strategy': 'random',
                'index': i,
                'weights': weights.tolist()
            }
            samples.append((weights, metadata))
        
        return samples
    
    def generate_extreme_modes(self, sigma: float = 2.0) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate shapes at ±sigma along each principal component.
        
        Useful for visualizing the range of each mode.
        
        Args:
            sigma: Number of standard deviations
            
        Returns:
            List of (weights, metadata) tuples (2*n_components shapes)
        """
        samples = []
        
        for k in range(self.n_components):
            # Positive extreme
            weights_pos = np.zeros(self.n_components)
            weights_pos[k] = sigma
            samples.append((weights_pos, {
                'strategy': 'extreme',
                'mode': k,
                'sigma': sigma,
                'direction': 'positive'
            }))
            
            # Negative extreme
            weights_neg = np.zeros(self.n_components)
            weights_neg[k] = -sigma
            samples.append((weights_neg, {
                'strategy': 'extreme',
                'mode': k,
                'sigma': -sigma,
                'direction': 'negative'
            }))
        
        return samples
    
    def generate_lhs(self, count: int) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate shapes using Latin Hypercube Sampling.
        
        Provides space-filling coverage of the latent space.
        Requires scipy.
        
        Args:
            count: Number of shapes to generate
            
        Returns:
            List of (weights, metadata) tuples
        """
        if not HAS_SCIPY:
            print("[Warning] scipy not available, falling back to random sampling")
            return self.generate_random(count)
        
        # LHS in [0, 1]^d
        sampler = qmc.LatinHypercube(d=self.n_components, seed=self.rng)
        lhs_samples = sampler.random(n=count)
        
        # Scale to [-sigma_range, +sigma_range]
        samples = []
        for i, sample in enumerate(lhs_samples):
            weights = 2 * self.sigma_range * sample - self.sigma_range
            
            metadata = {
                'strategy': 'lhs',
                'index': i,
                'weights': weights.tolist()
            }
            samples.append((weights, metadata))
        
        return samples
    
    def generate_grid(
        self,
        n_dims: int = 3,
        points_per_dim: int = 5
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate shapes on a regular grid in the first n dimensions.
        
        Args:
            n_dims: Number of dimensions to grid over (default: 3)
            points_per_dim: Points along each dimension (default: 5)
            
        Returns:
            List of (weights, metadata) tuples
        """
        n_dims = min(n_dims, self.n_components)
        
        # Create 1D grids
        grid_1d = np.linspace(-self.sigma_range, self.sigma_range, points_per_dim)
        
        # Create meshgrid
        grids = np.meshgrid(*[grid_1d for _ in range(n_dims)], indexing='ij')
        
        samples = []
        idx = 0
        
        # Iterate over all grid points
        for indices in np.ndindex(*[points_per_dim] * n_dims):
            weights = np.zeros(self.n_components)
            grid_coords = []
            
            for d, i in enumerate(indices):
                weights[d] = grids[d][indices]
                grid_coords.append(float(grids[d][indices]))
            
            metadata = {
                'strategy': 'grid',
                'index': idx,
                'grid_dims': n_dims,
                'grid_coords': grid_coords,
                'weights': weights.tolist()
            }
            samples.append((weights, metadata))
            idx += 1
        
        return samples
    
    def generate_interpolation(
        self,
        weights_start: np.ndarray,
        weights_end: np.ndarray,
        n_steps: int = 10
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate shapes by interpolating between two weight vectors.
        
        Args:
            weights_start: Starting weights
            weights_end: Ending weights
            n_steps: Number of interpolation steps
            
        Returns:
            List of (weights, metadata) tuples
        """
        samples = []
        
        for i in range(n_steps):
            t = i / (n_steps - 1) if n_steps > 1 else 0
            weights = (1 - t) * weights_start + t * weights_end
            
            metadata = {
                'strategy': 'interpolation',
                'index': i,
                't': t,
                'weights': weights.tolist()
            }
            samples.append((weights, metadata))
        
        return samples
    
    def weights_to_shape(self, weights: np.ndarray) -> np.ndarray:
        """Convert weight vector to point coordinates."""
        shape = generate_shape(self.model, weights)
        return shape_to_points(shape, destandardize=True)
    
    def save_shapes(
        self,
        samples: List[Tuple[np.ndarray, Dict]],
        output_dir: str,
        prefix: str = "synthetic"
    ) -> List[str]:
        """
        Generate and save shapes to OBJ files.
        
        Args:
            samples: List of (weights, metadata) from generate_* methods
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        for weights, metadata in samples:
            # Generate shape
            points = self.weights_to_shape(weights)
            
            # Build filename based on strategy
            strategy = metadata['strategy']
            if strategy == 'random':
                filename = f"{prefix}_random_{metadata['index']:04d}.obj"
            elif strategy == 'extreme':
                direction = 'pos' if metadata['sigma'] > 0 else 'neg'
                filename = f"{prefix}_mode{metadata['mode']:02d}_{direction}.obj"
            elif strategy == 'lhs':
                filename = f"{prefix}_lhs_{metadata['index']:04d}.obj"
            elif strategy == 'grid':
                coords_str = '_'.join(f"{c:.2f}" for c in metadata['grid_coords'])
                filename = f"{prefix}_grid_{metadata['index']:04d}_{coords_str}.obj"
            elif strategy == 'interpolation':
                filename = f"{prefix}_interp_{metadata['index']:04d}.obj"
            else:
                filename = f"{prefix}_{metadata.get('index', 0):04d}.obj"
            
            filepath = os.path.join(output_dir, filename)
            
            # Create header with metadata
            header = f"Synthetic femur | strategy={strategy}"
            if 'weights' in metadata:
                # Summarize weights
                w = metadata['weights'][:5]
                w_str = ', '.join(f"{x:.3f}" for x in w)
                header += f" | PC[0:5]=[{w_str}...]"
            
            # Save OBJ
            save_shape_to_obj(
                filepath,
                points,
                self.template_faces,
                self.template_normals if len(self.template_normals) > 0 else None,
                header
            )
            
            saved_files.append(filepath)
        
        return saved_files


# =============================================================================
# Metadata Export
# =============================================================================

def save_generation_metadata(
    samples: List[Tuple[np.ndarray, Dict]],
    output_path: str
) -> None:
    """
    Save generation metadata to JSON for reproducibility.
    
    Args:
        samples: List of (weights, metadata)
        output_path: Path to JSON file
    """
    import json
    
    records = []
    for weights, metadata in samples:
        record = metadata.copy()
        if 'weights' not in record:
            record['weights'] = weights.tolist()
        records.append(record)
    
    with open(output_path, 'w') as f:
        json.dump({
            'count': len(records),
            'samples': records
        }, f, indent=2)
    
    print(f"[Metadata] Saved to {output_path}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic femur shapes from PCA model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 random shapes
  python synthetic_data_generator.py --model model/pca_femur_model.bin \\
      --template ../data/training/L_Femur_11_DECIM.obj.FINAL.obj \\
      --output ../data/synthetic --count 100 --strategy random

  # Generate extreme mode variations (±2σ for each PC)
  python synthetic_data_generator.py --model model/pca_femur_model.bin \\
      --template ../data/training/L_Femur_11_DECIM.obj.FINAL.obj \\
      --output ../data/synthetic/extremes --strategy extreme --sigma 2.0

  # Latin Hypercube Sampling for comprehensive coverage
  python synthetic_data_generator.py --model model/pca_femur_model.bin \\
      --template ../data/training/L_Femur_11_DECIM.obj.FINAL.obj \\
      --output ../data/synthetic/lhs --count 50 --strategy lhs

  # Grid sampling in first 3 PCs (5^3 = 125 shapes)
  python synthetic_data_generator.py --model model/pca_femur_model.bin \\
      --template ../data/training/L_Femur_11_DECIM.obj.FINAL.obj \\
      --output ../data/synthetic/grid --strategy grid --grid-dims 3 --grid-points 5
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                        help='Path to PCA model binary file')
    parser.add_argument('--template', '-t', required=True,
                        help='Path to template OBJ file for face connectivity')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for generated shapes')
    parser.add_argument('--count', '-n', type=int, default=100,
                        help='Number of shapes to generate (default: 100)')
    parser.add_argument('--strategy', '-s', 
                        choices=['random', 'extreme', 'lhs', 'grid', 'all'],
                        default='random',
                        help='Sampling strategy (default: random)')
    parser.add_argument('--components', '-k', type=int, default=None,
                        help='Number of PCs to use (default: all)')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Sigma range for sampling (default: 3.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--grid-dims', type=int, default=3,
                        help='Number of dimensions for grid sampling (default: 3)')
    parser.add_argument('--grid-points', type=int, default=5,
                        help='Points per dimension for grid (default: 5)')
    parser.add_argument('--prefix', default='synthetic',
                        help='Filename prefix (default: synthetic)')
    parser.add_argument('--save-metadata', action='store_true',
                        help='Save generation metadata to JSON')
    
    args = parser.parse_args()
    
    # Load model
    print(f"\n=== Synthetic Data Generator ===\n")
    print(f"Loading PCA model: {args.model}")
    model = load_pca_model(args.model)
    print(f"  Dimensions: D={model.n_dimensions}")
    print(f"  Training samples: N={model.n_samples}")
    print(f"  Components: K={model.n_components}")
    
    # Variance info
    var_explained = 100 * np.cumsum(model.variances) / model.total_variance
    print(f"\n  Variance explained:")
    for k in [1, 3, 5, 7, 10]:
        if k <= model.n_components:
            print(f"    PC1-{k}: {var_explained[k-1]:.1f}%")
    
    # Initialize generator
    print(f"\nInitializing generator...")
    generator = SyntheticGenerator(
        model=model,
        template_path=args.template,
        n_components=args.components,
        sigma_range=args.sigma,
        seed=args.seed
    )
    
    # Generate samples based on strategy
    all_samples = []
    
    if args.strategy == 'random' or args.strategy == 'all':
        print(f"\n[Strategy: Random] Generating {args.count} samples...")
        samples = generator.generate_random(args.count)
        all_samples.extend(samples)
        
        output_dir = args.output if args.strategy == 'random' else os.path.join(args.output, 'random')
        files = generator.save_shapes(samples, output_dir, f"{args.prefix}_random")
        print(f"  Saved {len(files)} shapes to {output_dir}")
    
    if args.strategy == 'extreme' or args.strategy == 'all':
        print(f"\n[Strategy: Extreme] Generating mode variations at ±{args.sigma}σ...")
        samples = generator.generate_extreme_modes(sigma=args.sigma)
        all_samples.extend(samples)
        
        output_dir = args.output if args.strategy == 'extreme' else os.path.join(args.output, 'extreme')
        files = generator.save_shapes(samples, output_dir, f"{args.prefix}_extreme")
        print(f"  Saved {len(files)} shapes to {output_dir}")
    
    if args.strategy == 'lhs' or args.strategy == 'all':
        print(f"\n[Strategy: LHS] Generating {args.count} Latin Hypercube samples...")
        samples = generator.generate_lhs(args.count)
        all_samples.extend(samples)
        
        output_dir = args.output if args.strategy == 'lhs' else os.path.join(args.output, 'lhs')
        files = generator.save_shapes(samples, output_dir, f"{args.prefix}_lhs")
        print(f"  Saved {len(files)} shapes to {output_dir}")
    
    if args.strategy == 'grid' or args.strategy == 'all':
        total_grid = args.grid_points ** args.grid_dims
        print(f"\n[Strategy: Grid] Generating {total_grid} grid samples "
              f"({args.grid_dims} dims × {args.grid_points} points)...")
        samples = generator.generate_grid(
            n_dims=args.grid_dims,
            points_per_dim=args.grid_points
        )
        all_samples.extend(samples)
        
        output_dir = args.output if args.strategy == 'grid' else os.path.join(args.output, 'grid')
        files = generator.save_shapes(samples, output_dir, f"{args.prefix}_grid")
        print(f"  Saved {len(files)} shapes to {output_dir}")
    
    # Save metadata if requested
    if args.save_metadata:
        metadata_path = os.path.join(args.output, 'generation_metadata.json')
        save_generation_metadata(all_samples, metadata_path)
    
    print(f"\n=== Generation Complete ===")
    print(f"Total shapes generated: {len(all_samples)}")
    print(f"Output directory: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
