"""
LDDMM Configuration Module.

Provides centralized configuration for LDDMM registration, atlas building,
and tangent PCA operations.
"""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class LDDMMConfig:
    """Configuration for LDDMM registration.

    This dataclass encapsulates all parameters needed for LDDMM geodesic
    shooting. The parameters control the trade-off between deformation
    smoothness (regularization) and data fidelity (matching accuracy).

    Attributes:
        n_steps: Number of time discretization steps for geodesic integration.
            - n_steps=1: Linear deformation (NOT true LDDMM)
            - n_steps≥5: True LDDMM with geodesic shooting
            Higher values give more accurate geodesics but slower computation.

        kernel: RKHS kernel type defining the deformation regularity.
            - "gaussian": Standard choice, smooth deformations
            - "cauchy": Heavier tails, allows more local variation

        scale: Kernel bandwidth σ in mm. Controls spatial correlation:
            - Small σ: Local deformations, points move independently
            - Large σ: Global deformations, nearby points move together
            Rule of thumb: 10-20% of shape bounding box diagonal.

        regularization_weight: Weight λ for deformation energy penalty.
            Higher values enforce smoother deformations at the cost of
            matching accuracy. Typical range: 0.001 to 0.1.

        n_iter: Maximum optimizer iterations for registration.

        device: Compute device for PyTorch operations.
            - "auto": Select CUDA if available, else CPU
            - "cuda" or "cuda:0": Force GPU
            - "cpu": Force CPU

    Example:
        >>> config = LDDMMConfig(scale=15.0, n_steps=10)
        >>> registration = LDDMMRegistration(config)
    """

    n_steps: int = 5
    kernel: Literal["gaussian", "cauchy"] = "gaussian"
    scale: float = 10.0
    regularization_weight: float = 0.01
    n_iter: int = 100
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate configuration and resolve 'auto' device."""
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.regularization_weight < 0:
            raise ValueError("regularization_weight must be non-negative")
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def for_femurs(cls) -> "LDDMMConfig":
        """Preset configuration for femur landmarks.

        Tuned for femur point clouds with ~100mm bounding box diagonal.
        Uses moderate regularization to capture anatomical variation.

        Returns:
            LDDMMConfig configured for femur analysis.
        """
        return cls(
            n_steps=5,
            kernel="gaussian",
            scale=15.0,
            regularization_weight=0.01,
            n_iter=100,
        )

    @classmethod
    def high_precision(cls) -> "LDDMMConfig":
        """Preset for higher accuracy at the cost of speed.

        Uses more integration steps and optimizer iterations.
        Useful for final analysis or when accuracy is critical.

        Returns:
            LDDMMConfig with high-precision settings.
        """
        return cls(
            n_steps=10,
            kernel="gaussian",
            scale=15.0,
            regularization_weight=0.005,
            n_iter=200,
        )

    @classmethod
    def fast(cls) -> "LDDMMConfig":
        """Preset for quick exploratory analysis.

        Trades accuracy for speed. Useful for prototyping or
        large datasets where exact geodesics are less critical.

        Returns:
            LDDMMConfig with fast settings.
        """
        return cls(
            n_steps=3,
            kernel="gaussian",
            scale=20.0,
            regularization_weight=0.05,
            n_iter=50,
        )
