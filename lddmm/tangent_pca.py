"""
Tangent PCA for LDDMM Shape Analysis.

Performs Principal Component Analysis on initial momenta in the tangent
space at the atlas, enabling statistical analysis and shape synthesis
along principal geodesics.

The tangent space at the atlas is a linear space where standard PCA
applies. Each principal component represents a direction of maximum
variance in shape space, corresponding to a mode of anatomical variation.

This module uses true LDDMM throughout:
- Projection uses LDDMM registration to compute the log map
- Synthesis uses geodesic shooting for the exponential map

Example:
    >>> from lddmm import TangentPCA, AtlasBuilder
    >>> builder = AtlasBuilder()
    >>> result = builder.build(shapes)
    >>> pca = TangentPCA(n_components=10)
    >>> pca.fit(result.atlas, result.momenta)
    >>> new_shape = pca.synthesize_shape([1.0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0])
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import json

import numpy as np

from .config import LDDMMConfig
from .registration import LDDMMRegistration


class TangentPCA:
    """Principal Component Analysis in the tangent space at the atlas.

    Performs SVD-based PCA on the initial momenta (log maps) from the
    atlas to each training shape. The principal components represent
    directions of maximum variance in the shape manifold.

    This implementation uses true LDDMM:
    - `project()`: Computes log map via LDDMM registration
    - `synthesize_shape()`: Uses geodesic shooting for exponential map

    Attributes:
        n_components: Number of principal components to retain.
        config: LDDMM configuration for registration/shooting.
        atlas: The mean shape (N, 3).
        mean_momentum: Mean of all training momenta (N, 3).
        components: Principal components (n_components, N, 3).
        eigenvalues: Variance explained by each component.
        explained_variance_ratio: Fraction of variance per component.

    Example:
        >>> pca = TangentPCA(n_components=10)
        >>> pca.fit(atlas, momenta)
        >>> coeffs = pca.project(new_shape)
        >>> reconstructed = pca.synthesize_shape(coeffs)
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        config: Optional[LDDMMConfig] = None,
    ) -> None:
        """Initialize Tangent PCA.

        Args:
            n_components: Number of components to keep. If None, keeps all.
            config: LDDMM configuration for registration/shooting.
        """
        self.n_components = n_components
        self.config = config or LDDMMConfig()
        self._registration: Optional[LDDMMRegistration] = None

        # Fitted parameters (populated by fit())
        self.atlas: Optional[np.ndarray] = None
        self.mean_momentum: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.singular_values: Optional[np.ndarray] = None

        # Data dimensions
        self._n_samples: int = 0
        self._n_points: int = 0

    @property
    def n_samples_(self) -> int:
        """Number of training samples."""
        return self._n_samples

    @property
    def n_points_(self) -> int:
        """Number of points per shape."""
        return self._n_points

    @property
    def n_components_(self) -> int:
        """Number of retained components."""
        return self.n_components if self.n_components else 0

    def _get_registration(self) -> LDDMMRegistration:
        """Get or create the LDDMM registration object."""
        if self._registration is None:
            self._registration = LDDMMRegistration(self.config)
        return self._registration

    def fit(
        self, atlas: np.ndarray, momenta: np.ndarray
    ) -> "TangentPCA":
        """Fit Tangent PCA on initial momenta.

        Args:
            atlas: Mean shape (N, 3).
            momenta: Initial momenta array (K, N, 3) where K is the number
                of training shapes.

        Returns:
            self (fitted model).

        Raises:
            ValueError: If momenta dimensions are inconsistent.
        """
        self.atlas = atlas.copy()

        # Handle both list and array inputs
        if isinstance(momenta, list):
            momenta = np.stack(momenta, axis=0)

        if momenta.ndim != 3 or momenta.shape[2] != 3:
            raise ValueError(
                f"Expected momenta shape (K, N, 3), got {momenta.shape}"
            )

        self._n_samples = momenta.shape[0]
        self._n_points = momenta.shape[1]

        if atlas.shape != (self._n_points, 3):
            raise ValueError(
                f"Atlas shape {atlas.shape} inconsistent with "
                f"momenta shape {momenta.shape}"
            )

        # Compute mean momentum
        self.mean_momentum = momenta.mean(axis=0)  # (N, 3)

        # Center momenta
        momenta_centered = momenta - self.mean_momentum[np.newaxis, :, :]

        # Flatten: (K, N*3)
        K = self._n_samples
        D = self._n_points * 3
        momenta_flat = momenta_centered.reshape(K, D)

        # SVD: M = U @ S @ Vt
        U, S, Vt = np.linalg.svd(momenta_flat, full_matrices=False)

        # Determine number of components
        max_components = min(K, D)
        if self.n_components is None:
            self.n_components = max_components
        else:
            self.n_components = min(self.n_components, max_components)

        # Store results
        self.singular_values = S[: self.n_components]
        self.eigenvalues = S[: self.n_components] ** 2 / (K - 1)

        total_variance = np.sum(S**2) / (K - 1)
        self.explained_variance_ratio = self.eigenvalues / total_variance

        # Principal components: reshape Vt rows to (N, 3)
        self.components = Vt[: self.n_components].reshape(
            self.n_components, self._n_points, 3
        )

        print(f"[TangentPCA] Fitted:")
        print(f"  Samples: {self._n_samples}")
        print(f"  Points per shape: {self._n_points}")
        print(f"  Components kept: {self.n_components}")
        print(
            f"  Explained variance (top 5): "
            f"{self.explained_variance_ratio[:5].round(4)}"
        )
        print(
            f"  Cumulative (top 5): "
            f"{np.cumsum(self.explained_variance_ratio[:5]).round(4)}"
        )

        return self

    def project(self, shape: np.ndarray) -> np.ndarray:
        """Project a shape onto the principal components using true LDDMM.

        Computes the log map from atlas to shape via LDDMM registration,
        then projects the resulting momentum onto the principal component basis.

        Args:
            shape: Shape to project (N, 3).

        Returns:
            Coefficients (n_components,) in the PCA basis.

        Raises:
            RuntimeError: If model is not fitted.
        """
        self._check_fitted()

        # True LDDMM log map via registration
        registration = self._get_registration()
        momentum = registration.compute_momentum(self.atlas, shape)

        # Center
        momentum_centered = momentum - self.mean_momentum

        # Project onto components
        momentum_flat = momentum_centered.flatten()
        components_flat = self.components.reshape(self.n_components, -1)

        return momentum_flat @ components_flat.T

    def synthesize_shape(
        self, coefficients: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        """Synthesize a shape from PCA coefficients using geodesic shooting.

        Reconstructs the momentum from coefficients, then applies the
        exponential map via geodesic shooting to produce a shape.

        Args:
            coefficients: PCA coefficients (n_components,) or list.

        Returns:
            Synthesized shape (N, 3).

        Raises:
            RuntimeError: If model is not fitted.
        """
        self._check_fitted()

        coefficients = np.asarray(coefficients)
        if coefficients.shape[0] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} coefficients, "
                f"got {coefficients.shape[0]}"
            )

        # Reconstruct momentum
        components_flat = self.components.reshape(self.n_components, -1)
        momentum_flat = coefficients @ components_flat
        momentum = momentum_flat.reshape(self._n_points, 3)

        # Add mean momentum
        momentum = momentum + self.mean_momentum

        # True LDDMM exponential map via geodesic shooting
        registration = self._get_registration()
        return registration.shoot(self.atlas, momentum)

    def synthesize_along_mode(
        self, mode: int, t_values: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        """Synthesize shapes along a principal mode.

        Generates shapes by varying a single coefficient while keeping
        others at zero.

        Args:
            mode: Principal component index (0-based).
            t_values: Values in units of standard deviation.

        Returns:
            Array of shapes (len(t_values), N, 3).

        Raises:
            ValueError: If mode index is out of range.
        """
        self._check_fitted()

        if mode < 0 or mode >= self.n_components:
            raise ValueError(
                f"Mode {mode} not available (have {self.n_components} components)"
            )

        t_values = np.asarray(t_values)
        std = np.sqrt(self.eigenvalues[mode])

        shapes = []
        for t in t_values:
            # Coefficient vector with only mode-th entry nonzero
            coefficients = np.zeros(self.n_components)
            coefficients[mode] = t * std

            shape = self.synthesize_shape(coefficients)
            shapes.append(shape)

        return np.stack(shapes, axis=0)

    def get_mode_extremes(
        self, mode: int, n_std: float = 2.0, n_steps: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get shapes at extremes of a principal mode.

        Args:
            mode: Mode index (0-based).
            n_std: Number of standard deviations for extremes.
            n_steps: Number of steps from -n_std to +n_std.

        Returns:
            Tuple of (shapes, t_values):
                - shapes: (n_steps, N, 3)
                - t_values: (n_steps,)
        """
        t_values = np.linspace(-n_std, n_std, n_steps)
        shapes = self.synthesize_along_mode(mode, t_values)
        return shapes, t_values

    def reconstruct(
        self, shape: np.ndarray, n_components: Optional[int] = None
    ) -> np.ndarray:
        """Reconstruct a shape using PCA projection.

        Projects the shape to the latent space and reconstructs using
        the specified number of components.

        Args:
            shape: Input shape (N, 3).
            n_components: Number of components to use (default: all).

        Returns:
            Reconstructed shape (N, 3).
        """
        self._check_fitted()

        if n_components is None:
            n_components = self.n_components
        n_components = min(n_components, self.n_components)

        # Project to get coefficients
        coefficients = self.project(shape)

        # Zero out higher components
        coefficients_truncated = np.zeros(self.n_components)
        coefficients_truncated[:n_components] = coefficients[:n_components]

        return self.synthesize_shape(coefficients_truncated)

    def save(self, output_dir: str) -> None:
        """Save fitted PCA model to a single NPZ file.

        Creates `tangent_pca.npz` containing:
        - mean_momentum: Mean of training momenta (N, 3)
        - components: Principal components (n_components, N, 3)
        - eigenvalues: Variance per component
        - explained_variance_ratio: Fraction of variance per component
        - n_components, n_samples, n_points: Metadata

        Note: The atlas is NOT duplicated here; it's stored in atlas.npz.
        When loading, the atlas is read from atlas.npz.

        Args:
            output_dir: Directory to save the file.
        """
        self._check_fitted()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path / "tangent_pca.npz",
            mean_momentum=self.mean_momentum,
            components=self.components,
            eigenvalues=self.eigenvalues,
            explained_variance_ratio=self.explained_variance_ratio,
            n_components=np.array(self.n_components),
            n_samples=np.array(self._n_samples),
            n_points=np.array(self._n_points),
        )

        print(f"[TangentPCA] Saved to {output_path / 'tangent_pca.npz'}")

    @classmethod
    def load(cls, output_dir: str, config: Optional[LDDMMConfig] = None) -> "TangentPCA":
        """Load fitted PCA model from NPZ file.

        Loads tangent_pca.npz and reads atlas from atlas.npz in the same directory.

        Args:
            output_dir: Directory containing the npz files.
            config: LDDMM configuration for registration/shooting.

        Returns:
            TangentPCA with loaded parameters.
        """
        output_path = Path(output_dir)
        pca = cls(config=config)

        # Try new NPZ format first
        npz_file = output_path / "tangent_pca.npz"
        if npz_file.exists():
            data = np.load(npz_file)
            pca.mean_momentum = data["mean_momentum"]
            pca.components = data["components"]
            pca.eigenvalues = data["eigenvalues"]
            pca.explained_variance_ratio = data["explained_variance_ratio"]
            pca.n_components = int(data["n_components"])
            pca._n_samples = int(data["n_samples"])
            pca._n_points = int(data["n_points"])

            # Load atlas from atlas.npz
            atlas_file = output_path / "atlas.npz"
            if atlas_file.exists():
                atlas_data = np.load(atlas_file)
                pca.atlas = atlas_data["atlas"]
            else:
                raise FileNotFoundError(
                    f"atlas.npz not found in {output_path}. "
                    "TangentPCA requires the atlas file."
                )
            return pca

        # Legacy format fallback
        legacy_file = output_path / "tangent_pca_atlas.npy"
        if legacy_file.exists():
            pca.atlas = np.load(output_path / "tangent_pca_atlas.npy")
            pca.mean_momentum = np.load(output_path / "tangent_pca_mean_momentum.npy")
            pca.components = np.load(output_path / "tangent_pca_components.npy")
            pca.eigenvalues = np.load(output_path / "tangent_pca_eigenvalues.npy")
            pca.explained_variance_ratio = np.load(
                output_path / "tangent_pca_explained_variance.npy"
            )
            with open(output_path / "tangent_pca_metadata.json", "r") as f:
                metadata = json.load(f)
            pca.n_components = metadata["n_components"]
            pca._n_samples = metadata["n_samples"]
            pca._n_points = metadata["n_points"]
            return pca

        raise FileNotFoundError(f"No tangent PCA file found in {output_path}")

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if self.atlas is None or self.components is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
