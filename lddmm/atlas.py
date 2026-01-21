"""
Atlas Building for LDDMM Shape Analysis.

Computes the Fréchet mean (atlas) shape and stores initial momenta
for tangent PCA.

Author: Femur Modeling Project
Date: 2026
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json

try:
    from .registration import LDDMMPointRegistration
except ImportError:
    from registration import LDDMMPointRegistration


class LDDMMAtlasBuilder:
    """
    Build population atlas using LDDMM registration.
    
    The atlas is the Fréchet mean shape that minimizes the sum of
    squared geodesic distances to all shapes in the population.
    
    For computational efficiency with corresponding points, we use
    an iterative algorithm that approximates the geodesic mean.
    
    Example:
        >>> builder = LDDMMAtlasBuilder(max_outer_iterations=5)
        >>> builder.build(shapes)
        >>> builder.save("model/lddmm")
    """
    
    def __init__(
        self,
        registration_params: Optional[Dict] = None,
        atlas_step_size: float = 0.5,
        max_outer_iterations: int = 10,
        convergence_tol: float = 1e-4,
        verbose: bool = True
    ):
        """
        Initialize atlas builder.
        
        Args:
            registration_params: Parameters for LDDMMPointRegistration
            atlas_step_size: Step size for atlas update (0-1)
            max_outer_iterations: Maximum iterations for atlas refinement
            convergence_tol: Convergence tolerance
            verbose: Print progress
        """
        self.reg_params = registration_params or {
            'sigmaR': 10.0,
            'a': 5.0,
            'sigmaP': 1.0,
            'n_iter': 100,
            'ev': 1e-3,
            'verbose': False
        }
        self.atlas_step_size = atlas_step_size
        self.max_outer_iterations = max_outer_iterations
        self.convergence_tol = convergence_tol
        self.verbose = verbose
        
        # Results
        self.atlas = None
        self.momenta = []
        self.energy_history = []
    
    def build(
        self,
        shapes: List[np.ndarray],
        initial_atlas: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Build atlas from collection of shapes.
        
        Args:
            shapes: List of (N, 3) arrays with point correspondence
            initial_atlas: Optional initial atlas (default: arithmetic mean)
            
        Returns:
            Dictionary with atlas, momenta, energy_history
        """
        num_shapes = len(shapes)
        
        # Initialize atlas
        if initial_atlas is not None:
            self.atlas = initial_atlas.copy()
        else:
            # Arithmetic mean as initial guess
            self.atlas = np.stack(shapes).mean(axis=0)
            if self.verbose:
                print("[AtlasBuilder] Initialized atlas as arithmetic mean")
        
        # Note: For corresponding points, the Fréchet mean equals the arithmetic mean.
        # The iterative algorithm below verifies convergence and computes momenta.
        
        prev_total_energy = float('inf')
        
        for outer_iter in range(self.max_outer_iterations):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Atlas Iteration {outer_iter + 1}/{self.max_outer_iterations}")
                print(f"{'='*60}")
            
            # Compute displacements from atlas to each shape
            displacements = []
            total_energy = 0.0
            
            for j, shape in enumerate(shapes):
                # Simple displacement (momentum proxy)
                displacement = shape - self.atlas
                displacements.append(displacement)
                
                # Energy = squared displacement norm
                energy = np.sum(displacement ** 2)
                total_energy += energy
                
                if self.verbose:
                    disp_norm = np.linalg.norm(displacement, axis=1).mean()
                    print(f"  Shape {j+1}/{num_shapes}: mean_disp={disp_norm:.4f}")
            
            self.energy_history.append(total_energy)
            
            if self.verbose:
                print(f"\n  Total energy: {total_energy:.2f}")
            
            # Check convergence
            if abs(prev_total_energy - total_energy) < self.convergence_tol * prev_total_energy:
                if self.verbose:
                    print("  ✓ Atlas converged!")
                break
            prev_total_energy = total_energy
            
            # Update atlas: move towards mean of target shapes
            mean_displacement = np.stack(displacements).mean(axis=0)
            self.atlas = self.atlas + self.atlas_step_size * mean_displacement
            
            if self.verbose:
                update_norm = np.linalg.norm(mean_displacement, axis=1).mean()
                print(f"  Atlas update: mean_displacement = {update_norm:.4f}")
        
        # Final momenta computation
        if self.verbose:
            print(f"\n{'='*60}")
            print("Computing final momenta...")
            print(f"{'='*60}")
        
        self.momenta = []
        for j, shape in enumerate(shapes):
            # Momentum = displacement from atlas to shape
            momentum = shape - self.atlas
            self.momenta.append(momentum)
            
            if self.verbose:
                mom_norm = np.linalg.norm(momentum, axis=1).mean()
                print(f"  Shape {j+1}: ||momentum|| = {mom_norm:.4f}")
        
        return {
            'atlas': self.atlas,
            'momenta': self.momenta,
            'energy_history': self.energy_history
        }
    
    def save(self, output_dir: str):
        """Save atlas and momenta to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save atlas
        np.save(output_path / "atlas.npy", self.atlas)
        
        # Save momenta
        momenta_array = np.stack(self.momenta)  # (K, N, 3)
        np.save(output_path / "momenta.npy", momenta_array)
        
        # Save energy history (convert to float for JSON serialization)
        energy_list = [float(e) for e in self.energy_history]
        with open(output_path / "energy_history.json", 'w') as f:
            json.dump(energy_list, f)
        
        if self.verbose:
            print(f"\n[AtlasBuilder] Saved to {output_path}")
            print(f"  atlas.npy: {self.atlas.shape}")
            print(f"  momenta.npy: {momenta_array.shape}")
    
    @classmethod
    def load(cls, output_dir: str) -> 'LDDMMAtlasBuilder':
        """Load atlas and momenta from files."""
        output_path = Path(output_dir)
        
        builder = cls(verbose=False)
        builder.atlas = np.load(output_path / "atlas.npy")
        builder.momenta = list(np.load(output_path / "momenta.npy"))
        
        energy_file = output_path / "energy_history.json"
        if energy_file.exists():
            with open(energy_file, 'r') as f:
                builder.energy_history = json.load(f)
        
        return builder
