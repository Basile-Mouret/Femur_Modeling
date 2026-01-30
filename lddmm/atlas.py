"""
Atlas Building for LDDMM Shape Analysis.

Computes the population mean (Fréchet mean) shape and initial momenta for
tangent space analysis using true LDDMM geodesic averaging.

The atlas is computed via iterative geodesic averaging:

Initialize with arithmetic mean
Repeat until convergence:
    Compute log maps (momenta) from atlas to each shape via LDDMM registration
    Average momenta in tangent space
    Update atlas via exponential map (geodesic shooting) in direction of average momenta

Example:
    >>> from lddmm import AtlasBuilder, LDDMMConfig
    >>> builder = AtlasBuilder()
    >>> result = builder.build(shapes)
    >>> print(result.atlas.shape)  # (N, 3)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import time

import numpy as np

from .config import LDDMMConfig
from .registration import LDDMMRegistration


@dataclass
class AtlasResult:
    """Result of atlas building.

    Attributes:
        atlas: The Fréchet mean shape (N, 3).
        momenta: True LDDMM momenta from atlas to each shape (K, N, 3),
            computed via geodesic registration.
        convergence_history: Energy at each iteration.
    """

    atlas: np.ndarray
    momenta: np.ndarray
    convergence_history: List[float]


class AtlasBuilder:
    """Build population atlas (Fréchet mean) using true LDDMM.

    Computes the atlas via iterative geodesic averaging in the space of
    diffeomorphisms. The Fréchet mean minimizes the sum of squared geodesic
    distances to all shapes in the population.

    Algorithm:
        1. Initialize atlas as arithmetic mean of shapes
        2. For each iteration:
           a. Compute log maps: pᵢ = Log_μ(Sᵢ) via LDDMM registration
           b. Average in tangent space: p̄ = (1/K) Σᵢ pᵢ
           c. Update atlas: μ ← Exp_μ(α · p̄) via geodesic shooting
        3. Converge when energy change is below threshold

    Attributes:
        config: LDDMM configuration for geodesic computations.
        atlas: The computed Fréchet mean (after calling build()).
        momenta: True LDDMM momenta from atlas to each shape.
        convergence_history: Energy values during iteration.

    Example:
        >>> builder = AtlasBuilder()
        >>> result = builder.build(shapes)
        >>> builder.save("model/atlas")
    """

    def __init__(
        self,
        config: Optional[LDDMMConfig] = None,
        max_iterations: int = 10,
        convergence_tol: float = 1e-4,
        step_size: float = 0.5,
        verbose: bool = True,
    ) -> None:
        """Initialize atlas builder.

        Args:
            config: LDDMM configuration for registration.
            max_iterations: Maximum geodesic averaging iterations.
            convergence_tol: Relative energy change threshold for convergence.
            step_size: Step size for atlas update (0 < step_size ≤ 1).
            verbose: Print progress information.
        """
        self.config = config or LDDMMConfig()
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.step_size = step_size
        self.verbose = verbose

        # Results (populated by build())
        self.atlas: Optional[np.ndarray] = None
        self.momenta: Optional[np.ndarray] = None
        self.convergence_history: List[float] = []

    def build(self, shapes: List[np.ndarray]) -> AtlasResult:
        """Build atlas from a collection of shapes using geodesic averaging.

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
            print(f"  Points per shape: {n_points}")
            print(f"  Max atlas iterations: {self.max_iterations}")

        registration = LDDMMRegistration(self.config)

        # Initialize with arithmetic mean
        shapes_array = np.stack(shapes, axis=0)
        self.atlas = shapes_array.mean(axis=0)

        prev_energy = float("inf")
        self.convergence_history = []

        for iteration in range(self.max_iterations):
            iteration_start = time.time()
            
            if self.verbose:
                print(f"\n  === Atlas iteration {iteration + 1}/{self.max_iterations} ===")

            # Compute momenta (log maps) from atlas to each shape
            momenta_list = []
            for i, shape in enumerate(shapes):
                if self.verbose:
                    print(f"    [Registration {i + 1}/{len(shapes)}]")
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
                    iteration_time = time.time() - iteration_start
                    print(f"  Converged (relative change: {rel_change:.2e})")
                    print(f"  Iteration time: {iteration_time:.2f}s")
                break

            prev_energy = energy

            # Update atlas: shoot along mean momentum
            mean_momentum = self.momenta.mean(axis=0)
            scaled_momentum = self.step_size * mean_momentum

            # Geodesic update via exponential map
            self.atlas = registration.shoot(self.atlas, scaled_momentum)

            if self.verbose:
                update_norm = np.linalg.norm(scaled_momentum)
                iteration_time = time.time() - iteration_start
                print(f"  Atlas update norm: {update_norm:.4f}")
                print(f"  Iteration time: {iteration_time:.2f}s")
        else:
            # Loop completed without convergence (hit max_iterations)
            # Need to recompute momenta at final atlas position since
            # atlas was updated after the last momenta computation
            if self.verbose:
                print(f"\n  Max iterations reached, computing final momenta...")
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
        """Save atlas and momenta to a single NPZ file.

        Creates `atlas.npz` containing:
        - atlas: The Fréchet mean shape (N, 3)
        - momenta: Initial momenta from atlas to each training shape (K, N, 3)
        - convergence_history: Energy at each iteration
        - n_shapes: Number of training shapes
        - n_points: Number of points per shape

        Args:
            output_dir: Directory to save the file.
        """
        if self.atlas is None or self.momenta is None:
            raise RuntimeError("No atlas to save. Call build() first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path / "atlas.npz",
            atlas=self.atlas,
            momenta=self.momenta,
            convergence_history=np.array(self.convergence_history),
            n_shapes=np.array(self.momenta.shape[0]),
            n_points=np.array(self.atlas.shape[0]),
        )

        if self.verbose:
            print(f"[AtlasBuilder] Saved to {output_path / 'atlas.npz'}")

    @classmethod
    def load(cls, output_dir: str) -> "AtlasBuilder":
        """Load atlas and momenta from NPZ file.

        Args:
            output_dir: Directory containing atlas.npz.

        Returns:
            AtlasBuilder with loaded atlas and momenta.
        """
        output_path = Path(output_dir)

        # Support both new NPZ format and legacy format
        npz_file = output_path / "atlas.npz"
        if npz_file.exists():
            data = np.load(npz_file)
            builder = cls(verbose=False)
            builder.atlas = data["atlas"]
            builder.momenta = data["momenta"]
            builder.convergence_history = data["convergence_history"].tolist()
            return builder

        # Legacy format fallback
        legacy_atlas = output_path / "atlas.npy"
        if legacy_atlas.exists():
            builder = cls(verbose=False)
            builder.atlas = np.load(output_path / "atlas.npy")
            builder.momenta = np.load(output_path / "momenta.npy")
            metadata_file = output_path / "atlas_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                builder.convergence_history = metadata.get("convergence_history", [])
            return builder

        raise FileNotFoundError(f"No atlas file found in {output_path}")
