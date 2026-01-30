"""
LDDMM Registration using scikit-shapes.

Provides LDDMM (Large Deformation Diffeomorphic Metric Mapping) registration
for point clouds with established point correspondence.

This module implements Log map computation in Riemannian manifold of shapes, via iterative
optimization and geodesic shooting (Exp map). The key output is the initial momentum
field that generates the diffeomorphic transformation between source and target shapes.


Example:
    >>> from lddmm import LDDMMRegistration, LDDMMConfig
    >>> config = LDDMMConfig.for_femurs()
    >>> registration = LDDMMRegistration(config)
    >>> result = registration.register(source, target)
    >>> print(result.momentum.shape)  # (N, 3)
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

try:
    import skshapes as sks
    from skshapes.morphing import extrinsic_deformation

    # Fix type annotation bug in scikit-shapes 0.3
    # torchdiffeq passes t as tensor but ODEModule.__call__ expects float
    def _patched_odemodule_call(self, t, y):
        """Call the ODE function, converting tensor t to float if needed."""
        t_float = float(t) if isinstance(t, torch.Tensor) else t
        return self.func(t_float, y)

    extrinsic_deformation.ODEModule.__call__ = _patched_odemodule_call

    SKSHAPES_AVAILABLE = True
except ImportError:
    SKSHAPES_AVAILABLE = False
    sks = None

from .config import LDDMMConfig


@dataclass
class RegistrationResult:
    """Result of LDDMM registration.

    Attributes:
        momentum: Initial momentum (N, 3) that generates the geodesic from
            source to target.
        transformed: The deformed source shape (N, 3), should match target.
        path: List of intermediate shapes along the geodesic (if available).
        energy: Deformation energy (regularization term value).
        success: Whether registration converged successfully.
    """

    momentum: np.ndarray
    transformed: np.ndarray
    path: Optional[List[np.ndarray]] = None
    energy: float = 0.0
    success: bool = True


class LDDMMRegistration:
    """LDDMM registration for point clouds with correspondence.

    The transformation is
    parametrized by an initial momentum field at the source points.

    For shapes with known point correspondence, we use L2Loss which
    measures the squared distance between corresponding points.

    Attributes:
        config: LDDMM configuration parameters.

    Example:
        >>> config = LDDMMConfig(n_steps=5, scale=15.0)
        >>> reg = LDDMMRegistration(config)
        >>> result = reg.register(source_points, target_points)
        >>> momentum = result.momentum  # Use for tangent PCA
    """

    def __init__(self, config: Optional[LDDMMConfig] = None) -> None:
        """Initialize LDDMM registration.

        Args:
            config: LDDMM configuration. If None, uses default settings.

        Raises:
            ImportError: If scikit-shapes is not installed.
        """
        if not SKSHAPES_AVAILABLE:
            raise ImportError(
                "scikit-shapes is required for LDDMM registration. "
                "Install with: pip install skshapes"
            )

        self.config = config or LDDMMConfig()
        self._model: Optional[sks.ExtrinsicDeformation] = None
        self._registration: Optional[sks.Registration] = None

    def _build_model(self) -> sks.ExtrinsicDeformation:
        """Build the scikit-shapes deformation model."""
        return sks.ExtrinsicDeformation(
            n_steps=self.config.n_steps,
            kernel=self.config.kernel,
            scale=self.config.scale,
            control_points=False,  # Use shape points directly as control points
        )

    def _build_registration(self, model: sks.ExtrinsicDeformation) -> sks.Registration:
        """Build the scikit-shapes registration object."""
        return sks.Registration(
            model=model,
            loss=sks.L2Loss(),  # L2 loss for corresponding points
            optimizer=sks.LBFGS(),
            n_iter=self.config.n_iter,
            regularization_weight=self.config.regularization_weight,
            verbose=self.config.verbose,
        )

    def register(
        self, source: np.ndarray, target: np.ndarray
    ) -> RegistrationResult:
        """Register source shape to target shape.

        Computes the LDDMM geodesic from source to target, returning the
        initial momentum that generates this transformation.

        Args:
            source: Source point cloud (N, 3).
            target: Target point cloud (N, 3). Must have same N as source
                (point correspondence required).

        Returns:
            RegistrationResult containing momentum, transformed shape, and
            deformation energy.

        Raises:
            ValueError: If source and target have different shapes.
        """
        if source.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: source {source.shape} vs target {target.shape}. "
                "Point correspondence requires equal point counts."
            )

        # Convert to PolyData
        source_poly = sks.PolyData(points=torch.tensor(source, dtype=torch.float32))
        target_poly = sks.PolyData(points=torch.tensor(target, dtype=torch.float32))

        # Move to device
        if self.config.device != "cpu":
            source_poly = source_poly.to(self.config.device)
            target_poly = target_poly.to(self.config.device)

        # Build fresh model and registration for each call
        # (ensures clean state)
        model = self._build_model()
        registration = self._build_registration(model)

        # Perform registration
        morphed = registration.fit_transform(source=source_poly, target=target_poly)

        # Extract momentum from the model parameter
        # In scikit-shapes, the parameter is the momentum field
        momentum = self._extract_momentum(registration, source_poly)

        # Extract path if available
        path = self._extract_path(registration)

        # Compute deformation energy
        energy = self._compute_energy(registration)

        # Get transformed points
        transformed = morphed.points.detach().cpu().numpy()

        return RegistrationResult(
            momentum=momentum,
            transformed=transformed,
            path=path,
            energy=energy,
            success=True,
        )

    def _extract_momentum(
        self, registration: sks.Registration, source: sks.PolyData
    ) -> np.ndarray:
        """Extract initial momentum from the fitted registration.

        The momentum is stored as the model parameter after fitting.
        """
        # The parameter_ attribute contains the fitted momentum
        if hasattr(registration, "parameter_") and registration.parameter_ is not None:
            return registration.parameter_.detach().cpu().numpy()

        raise RuntimeError(
            "Could not extract momentum from registration. "
            "The registration may have failed or the scikit-shapes API changed."
        )

    def _extract_path(self, registration: sks.Registration) -> Optional[List[np.ndarray]]:
        """Extract geodesic path from registration if available."""
        if hasattr(registration, "path_") and registration.path_ is not None:
            return [
                shape.points.detach().cpu().numpy()
                for shape in registration.path_
            ]
        return None

    def _compute_energy(self, registration: sks.Registration) -> float:
        """Compute deformation energy (regularization term)."""
        if hasattr(registration, "regularization_") and registration.regularization_ is not None:
            return float(registration.regularization_.detach().cpu())
        return 0.0

    def compute_momentum(
        self, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Compute initial momentum from source to target.

        Convenience method that performs registration and returns only
        the momentum field. This is the log map: Log_source(target).

        Args:
            source: Source point cloud (N, 3).
            target: Target point cloud (N, 3).

        Returns:
            Initial momentum (N, 3).
        """
        result = self.register(source, target)
        return result.momentum

    def shoot(
        self, source: np.ndarray, momentum: np.ndarray
    ) -> np.ndarray:
        """Shoot from source along momentum (exponential map).

        Computes: Exp_source(momentum) = geodesic endpoint.

        This is the inverse of compute_momentum (log map).

        Args:
            source: Source point cloud (N, 3).
            momentum: Initial momentum field (N, 3).

        Returns:
            Deformed shape (N, 3).
        """
        if source.shape != momentum.shape:
            raise ValueError(
                f"Shape mismatch: source {source.shape} vs momentum {momentum.shape}"
            )

        # Convert to PolyData
        source_poly = sks.PolyData(points=torch.tensor(source, dtype=torch.float32))
        momentum_tensor = torch.tensor(momentum, dtype=torch.float32)

        if self.config.device != "cpu":
            source_poly = source_poly.to(self.config.device)
            momentum_tensor = momentum_tensor.to(self.config.device)

        # Build model and apply morphing
        model = self._build_model()

        # Use model's morph method with the momentum as parameter
        morphed = model.morph(
            shape=source_poly,
            parameter=momentum_tensor,
        )
        return morphed.points.detach().cpu().numpy()
