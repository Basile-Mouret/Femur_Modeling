"""
Tangent PCA for LDDMM Shape Analysis.

Performs PCA on initial momenta in the tangent space at the atlas,
enabling shape synthesis along principal geodesics.

Author: Femur Modeling Project  
Date: 2026
"""

import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import json


class TangentPCA:
    """
    Principal Component Analysis in the tangent space of the atlas.
    
    Performs standard PCA on the initial momenta (displacements) that
    map the atlas to each shape. The principal components represent
    directions of maximum variance in shape space.
    
    Example:
        >>> pca = TangentPCA(n_components=10)
        >>> pca.fit(atlas, momenta)
        >>> shapes = pca.synthesize_along_mode(mode=0, t_values=[-2, -1, 0, 1, 2])
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize Tangent PCA.
        
        Args:
            n_components: Number of components to keep (None = all)
        """
        self.n_components = n_components
        
        # Fitted parameters
        self.atlas = None
        self.mean_momentum = None
        self.components = None       # (n_components, N, 3)
        self.eigenvalues = None      # (n_components,)
        self.explained_variance_ratio = None
        self.singular_values = None
        
        # Data info
        self.n_samples = None
        self.n_points = None
    
    def fit(
        self,
        atlas: np.ndarray,
        momenta: List[np.ndarray]
    ) -> 'TangentPCA':
        """
        Fit Tangent PCA on momenta.
        
        Args:
            atlas: (N, 3) atlas shape
            momenta: List of K (N, 3) momenta arrays
            
        Returns:
            self (fitted)
        """
        self.atlas = atlas.copy()
        self.n_samples = len(momenta)
        self.n_points = atlas.shape[0]
        
        # Stack momenta: (K, N, 3)
        M = np.stack(momenta, axis=0)
        
        # Compute mean momentum
        self.mean_momentum = M.mean(axis=0)
        
        # Center momenta
        M_centered = M - self.mean_momentum[np.newaxis, :, :]
        
        # Flatten: (K, N*3)
        K = self.n_samples
        D = self.n_points * 3
        M_flat = M_centered.reshape(K, D)
        
        # SVD: M_flat = U @ S @ Vt
        U, S, Vt = np.linalg.svd(M_flat, full_matrices=False)
        
        # Determine number of components
        max_components = min(K, D)
        if self.n_components is None:
            self.n_components = max_components
        else:
            self.n_components = min(self.n_components, max_components)
        
        # Store results
        self.singular_values = S[:self.n_components]
        self.eigenvalues = S[:self.n_components] ** 2 / (K - 1)
        
        total_variance = np.sum(S ** 2) / (K - 1)
        self.explained_variance_ratio = self.eigenvalues / total_variance
        
        # Principal components: reshape Vt rows to (N, 3)
        self.components = Vt[:self.n_components].reshape(
            self.n_components, self.n_points, 3
        )
        
        print(f"[TangentPCA] Fitted:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Points per shape: {self.n_points}")
        print(f"  Components kept: {self.n_components}")
        print(f"  Explained variance (top 5): {self.explained_variance_ratio[:5]}")
        print(f"  Cumulative (top 5): {np.cumsum(self.explained_variance_ratio[:5])}")
        
        return self
    
    def transform(self, momenta: List[np.ndarray]) -> np.ndarray:
        """
        Project momenta onto principal components.
        
        Args:
            momenta: List of K' (N, 3) momenta
            
        Returns:
            coefficients: (K', n_components) projection coefficients
        
        Raises:
            RuntimeError: If model is not fitted
        """
        if self.components is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        M = np.stack(momenta, axis=0)
        M_centered = M - self.mean_momentum[np.newaxis, :, :]
        M_flat = M_centered.reshape(M.shape[0], -1)
        
        components_flat = self.components.reshape(self.n_components, -1)
        
        return M_flat @ components_flat.T
    
    def project(self, shape: np.ndarray) -> np.ndarray:
        """
        Project a shape to the PCA coefficient space.
        
        First computes the momentum (log map) from atlas to shape,
        then projects the momentum onto the principal components.
        
        Args:
            shape: (N, 3) shape to project
            
        Returns:
            coefficients: (n_components,) PCA coefficients
            
        Raises:
            RuntimeError: If model is not fitted
        """
        if self.atlas is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Linearized log map: momentum = shape - atlas
        momentum = shape - self.atlas
        
        # Center and project
        momentum_centered = momentum - self.mean_momentum
        momentum_flat = momentum_centered.flatten()
        
        components_flat = self.components.reshape(self.n_components, -1)
        
        return momentum_flat @ components_flat.T
    
    def reconstruct(self, shape: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct a shape using PCA projection/reconstruction.
        
        Projects the shape to the latent space and reconstructs it
        using the specified number of components.
        
        Args:
            shape: (N, 3) shape to reconstruct
            n_components: Number of components to use (default: all)
            
        Returns:
            reconstructed: (N, 3) reconstructed shape
            
        Raises:
            RuntimeError: If model is not fitted
        """
        if self.atlas is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if n_components is None:
            n_components = self.n_components
        n_components = min(n_components, self.n_components)
        
        # Project to get all coefficients
        coefficients = self.project(shape)
        
        # Use only first n_components
        coefficients_truncated = np.zeros(self.n_components)
        coefficients_truncated[:n_components] = coefficients[:n_components]
        
        # Reconstruct
        return self.synthesize_shape(coefficients_truncated)
    
    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Reconstruct momenta from coefficients.
        
        Args:
            coefficients: (K', n_components) or (n_components,)
            
        Returns:
            momenta: (K', N, 3) or (N, 3) reconstructed momenta
        
        Raises:
            RuntimeError: If model is not fitted
        """
        if self.components is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, :]
        
        components_flat = self.components.reshape(self.n_components, -1)
        M_flat = coefficients @ components_flat
        
        M = M_flat.reshape(-1, self.n_points, 3)
        M = M + self.mean_momentum[np.newaxis, :, :]
        
        return M.squeeze()
    
    def synthesize_shape(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Synthesize a shape from PCA coefficients.
        
        Args:
            coefficients: (n_components,) array of coefficients
            
        Returns:
            shape: (N, 3) synthesized shape
        
        Raises:
            RuntimeError: If model is not fitted
        """
        if self.atlas is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        momentum = self.inverse_transform(coefficients)
        # Linearized exponential map: shape = atlas + momentum
        return self.atlas + momentum
    
    def synthesize_along_mode(
        self,
        mode: int,
        t_values: np.ndarray
    ) -> np.ndarray:
        """
        Synthesize shapes along a principal mode.
        
        Args:
            mode: Principal component index (0-based)
            t_values: Array of t values (in units of std deviation)
            
        Returns:
            shapes: (len(t_values), N, 3) array of shapes
        
        Raises:
            RuntimeError: If model is not fitted
            ValueError: If mode index is out of range
        """
        if self.components is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if mode < 0 or mode >= self.n_components:
            raise ValueError(f"Mode {mode} not available (have {self.n_components} components)")
        
        std = np.sqrt(self.eigenvalues[mode])
        
        shapes = []
        for t in t_values:
            # Momentum = mean + t * std * component
            momentum = self.mean_momentum + t * std * self.components[mode]
            # Shape = atlas + momentum (linearized exponential map)
            shape = self.atlas + momentum
            shapes.append(shape)
        
        return np.stack(shapes, axis=0)
    
    def get_mode_extremes(
        self,
        mode: int,
        n_std: float = 2.0,
        n_steps: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get shapes at extremes of a mode.
        
        Args:
            mode: Mode index
            n_std: Number of standard deviations
            n_steps: Number of steps from -n_std to +n_std
            
        Returns:
            shapes: (n_steps, N, 3)
            t_values: (n_steps,)
        """
        t_values = np.linspace(-n_std, n_std, n_steps)
        shapes = self.synthesize_along_mode(mode, t_values)
        return shapes, t_values
    
    def save(self, output_dir: str):
        """Save fitted PCA model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / "tangent_pca_atlas.npy", self.atlas)
        np.save(output_path / "tangent_pca_mean_momentum.npy", self.mean_momentum)
        np.save(output_path / "tangent_pca_components.npy", self.components)
        np.save(output_path / "tangent_pca_eigenvalues.npy", self.eigenvalues)
        np.save(output_path / "tangent_pca_explained_variance.npy", 
                self.explained_variance_ratio)
        
        metadata = {
            'n_components': int(self.n_components),
            'n_samples': int(self.n_samples),
            'n_points': int(self.n_points)
        }
        with open(output_path / "tangent_pca_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[TangentPCA] Saved to {output_path}")
    
    @classmethod
    def load(cls, output_dir: str) -> 'TangentPCA':
        """Load fitted PCA model."""
        output_path = Path(output_dir)
        
        pca = cls()
        pca.atlas = np.load(output_path / "tangent_pca_atlas.npy")
        pca.mean_momentum = np.load(output_path / "tangent_pca_mean_momentum.npy")
        pca.components = np.load(output_path / "tangent_pca_components.npy")
        pca.eigenvalues = np.load(output_path / "tangent_pca_eigenvalues.npy")
        pca.explained_variance_ratio = np.load(
            output_path / "tangent_pca_explained_variance.npy"
        )
        
        with open(output_path / "tangent_pca_metadata.json", 'r') as f:
            metadata = json.load(f)
        pca.n_components = metadata['n_components']
        pca.n_samples = metadata['n_samples']
        pca.n_points = metadata['n_points']
        
        return pca
