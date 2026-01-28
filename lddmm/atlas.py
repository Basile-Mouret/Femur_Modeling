"""
Atlas Building for LDDMM Shape Analysis.

Computes the population mean (atlas) shape and initial momenta for
tangent space analysis.

This module provides two methods for computing the atlas:
1. Arithmetic mean (default): Exact Fréchet mean for corresponding points
2. Geodesic mean: Iterative averaging using LDDMM (for soft correspondence)

Theory:
    See LDDMM_THEORY.md for mathematical background on:
    - Fréchet mean on shape manifolds
    - Why arithmetic mean equals geodesic mean for corresponding points

Example:
    >>> from lddmm import AtlasBuilder, LDDMMConfig
    >>> builder = AtlasBuilder()
    >>> result = builder.build(shapes)
    >>> print(result.atlas.shape)  # (N, 3)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional
import json

import numpy as np

from .config import LDDMMConfig
from .registration import LDDMMRegistration


@dataclass
class AtlasResult:
    """Result of atlas building.

    Attributes:
        atlas: The mean shape (N, 3).
        momenta: Initial momenta from atlas to each shape (K, N, 3).
        convergence_history: Energy at each iteration (for geodesic method).
    """

    atlas: np.ndarray
    momenta: np.ndarray
    convergence_history: List[float]


class AtlasBuilder:
    """Build population atlas (Fréchet mean) from a collection of shapes.

    The atlas is the shape that minimizes the sum of squared geodesic
    distances to all shapes in the population.

    Two methods are available:

    1. **Arithmetic mean** (default, `method='arithmetic'`):
       Simply averages the corresponding point positions. This is
       mathematically equivalent to the Fréchet mean when:
       - All shapes have established point correspondence
       - Points live in Euclidean space (ℝ³)

       Why? For corresponding points, the geodesic distance equals the
       Euclidean distance: d(S₁, S₂) = ‖S₁ - S₂‖_F. The minimizer of
       Σᵢ ‖μ - Sᵢ‖² is the arithmetic mean μ = (1/K) Σᵢ Sᵢ.

    2. **Geodesic mean** (`method='geodesic'`):
       Iteratively refines the atlas using geodesic averaging in tangent
       space. This is useful when:
       - Correspondence is uncertain or soft
       - Additional regularization is desired
       - Comparing to true LDDMM geodesic distances

    Attributes:
        config: LDDMM configuration for geodesic computations.
        method: 'arithmetic' or 'geodesic'.
        atlas: The computed atlas (after calling build()).
        momenta: Momenta from atlas to each shape.

    Example:
        >>> builder = AtlasBuilder(method='arithmetic')
        >>> result = builder.build(shapes)
        >>> builder.save("model/atlas")
    """

    def __init__(
        self,
        config: Optional[LDDMMConfig] = None,
        method: Literal["arithmetic", "geodesic"] = "arithmetic",
        max_iterations: int = 10,
        convergence_tol: float = 1e-4,
        step_size: float = 0.5,
        verbose: bool = True,
    ) -> None:
        """Initialize atlas builder.

        Args:
            config: LDDMM configuration for registration (geodesic method).
            method: Atlas computation method:
                - 'arithmetic': Direct averaging (fast, exact for correspondence)
                - 'geodesic': Iterative geodesic averaging (slower, more general)
            max_iterations: Maximum iterations for geodesic method.
            convergence_tol: Relative energy change threshold for convergence.
            step_size: Step size for geodesic atlas update (0 < step_size ≤ 1).
            verbose: Print progress information.
        """
        self.config = config or LDDMMConfig()
        self.method = method
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.step_size = step_size
        self.verbose = verbose

        # Results (populated by build())
        self.atlas: Optional[np.ndarray] = None
        self.momenta: Optional[np.ndarray] = None
        self.convergence_history: List[float] = []

    def build(self, shapes: List[np.ndarray]) -> AtlasResult:
        """Build atlas from a collection of shapes.

        Args:
            shapes: List of K shapes, each (N, 3) with point correspondence.

        Returns:
            AtlasResult containing atlas, momenta, and convergence history.

        Raises:
            ValueError: If shapes have inconsistent dimensions.
        """
        if len(shapes) < 2:
            raise ValueError("Need at least 2 shapes to build atlas")

        # Validate shapes
        n_points = shapes[0].shape[0]
        for i, shape in enumerate(shapes):
            if shape.shape != (n_points, 3):
                raise ValueError(
                    f"Shape {i} has dimensions {shape.shape}, "
                    f"expected ({n_points}, 3)"
                )

        if self.verbose:
            print(f"[AtlasBuilder] Building atlas from {len(shapes)} shapes")
            print(f"  Method: {self.method}")
            print(f"  Points per shape: {n_points}")

        if self.method == "arithmetic":
            return self._build_arithmetic(shapes)
        else:
            return self._build_geodesic(shapes)

    def _build_arithmetic(self, shapes: List[np.ndarray]) -> AtlasResult:
        """Build atlas using arithmetic mean.

        This is the exact Fréchet mean for shapes with point correspondence
        in Euclidean space. See class docstring for mathematical justification.
        """
        # Stack and compute mean
        shapes_array = np.stack(shapes, axis=0)  # (K, N, 3)
        self.atlas = shapes_array.mean(axis=0)  # (N, 3)

        # Compute momenta (displacements from atlas to each shape)
        # For corresponding points in Euclidean space, momentum = displacement
        self.momenta = shapes_array - self.atlas[np.newaxis, :, :]  # (K, N, 3)

        # No iterations for arithmetic mean
        self.convergence_history = [self._compute_total_energy(shapes)]

        if self.verbose:
            print(f"  Atlas computed (arithmetic mean)")
            print(f"  Total energy: {self.convergence_history[0]:.4f}")

        return AtlasResult(
            atlas=self.atlas,
            momenta=self.momenta,
            convergence_history=self.convergence_history,
        )

    def _build_geodesic(self, shapes: List[np.ndarray]) -> AtlasResult:
        """Build atlas using iterative geodesic averaging.

        Algorithm:
        1. Initialize atlas as arithmetic mean
        2. Repeat until convergence:
           a. Compute log maps (momenta) from atlas to each shape
           b. Average momenta in tangent space
           c. Update atlas by shooting along mean momentum
        """
        registration = LDDMMRegistration(self.config)

        # Initialize with arithmetic mean
        shapes_array = np.stack(shapes, axis=0)
        self.atlas = shapes_array.mean(axis=0)

        prev_energy = float("inf")
        self.convergence_history = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n  Iteration {iteration + 1}/{self.max_iterations}")

            # Compute momenta (log maps) from atlas to each shape
            momenta_list = []
            for i, shape in enumerate(shapes):
                momentum = registration.compute_momentum(self.atlas, shape)
                momenta_list.append(momentum)

            self.momenta = np.stack(momenta_list, axis=0)  # (K, N, 3)

            # Compute energy
            energy = self._compute_total_energy(shapes)
            self.convergence_history.append(energy)

            if self.verbose:
                print(f"  Energy: {energy:.4f}")

            # Check convergence
            rel_change = abs(prev_energy - energy) / (abs(prev_energy) + 1e-10)
            if rel_change < self.convergence_tol:
                if self.verbose:
                    print(f"  Converged (relative change: {rel_change:.2e})")
                break

            prev_energy = energy

            # Update atlas: shoot along mean momentum
            mean_momentum = self.momenta.mean(axis=0)
            scaled_momentum = self.step_size * mean_momentum

            # Geodesic update via exponential map
            self.atlas = registration.shoot(self.atlas, scaled_momentum)

            if self.verbose:
                update_norm = np.linalg.norm(scaled_momentum)
                print(f"  Atlas update norm: {update_norm:.4f}")

        # Final momenta computation at converged atlas
        momenta_list = []
        for shape in shapes:
            momentum = registration.compute_momentum(self.atlas, shape)
            momenta_list.append(momentum)
        self.momenta = np.stack(momenta_list, axis=0)

        if self.verbose:
            print(f"\n  Atlas building complete")
            print(f"  Final energy: {self.convergence_history[-1]:.4f}")

        return AtlasResult(
            atlas=self.atlas,
            momenta=self.momenta,
            convergence_history=self.convergence_history,
        )

    def _compute_total_energy(self, shapes: List[np.ndarray]) -> float:
        """Compute total squared distance from atlas to all shapes."""
        if self.atlas is None:
            return float("inf")

        total = 0.0
        for shape in shapes:
            diff = shape - self.atlas
            total += np.sum(diff**2)
        return total

    def save(self, output_dir: str) -> None:
        """Save atlas and momenta to files.

        Args:
            output_dir: Directory to save files.
        """
        if self.atlas is None or self.momenta is None:
            raise RuntimeError("No atlas to save. Call build() first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.save(output_path / "atlas.npy", self.atlas)
        np.save(output_path / "momenta.npy", self.momenta)

        metadata = {
            "method": self.method,
            "n_shapes": int(self.momenta.shape[0]),
            "n_points": int(self.atlas.shape[0]),
            "convergence_history": [float(e) for e in self.convergence_history],
        }
        with open(output_path / "atlas_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"[AtlasBuilder] Saved to {output_path}")

    @classmethod
    def load(cls, output_dir: str) -> "AtlasBuilder":
        """Load atlas and momenta from files.

        Args:
            output_dir: Directory containing saved files.

        Returns:
            AtlasBuilder with loaded atlas and momenta.
        """
        output_path = Path(output_dir)

        builder = cls(verbose=False)
        builder.atlas = np.load(output_path / "atlas.npy")
        builder.momenta = np.load(output_path / "momenta.npy")

        metadata_file = output_path / "atlas_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            builder.method = metadata.get("method", "arithmetic")
            builder.convergence_history = metadata.get("convergence_history", [])

        return builder
