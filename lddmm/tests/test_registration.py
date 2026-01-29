#!/usr/bin/env python3
"""Tests for LDDMM registration module."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from lddmm.config import LDDMMConfig
from lddmm.registration import LDDMMRegistration, RegistrationResult
from lddmm.femur_lddmm.data_loader import FemurDataLoader


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def synthetic_shapes():
    """Create small synthetic shapes for fast testing."""
    np.random.seed(42)
    n_points = 100  # Small point cloud for fast tests
    
    # Create base shape (sphere-ish)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 10.0  # radius in mm
    
    source = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi),
    ]).astype(np.float32)
    
    # Create target with small deformation
    target = source + np.random.randn(n_points, 3).astype(np.float32) * 0.5
    
    return source, target


@pytest.fixture
def source_target(synthetic_shapes):
    """Return a pair of shapes for registration."""
    return synthetic_shapes


@pytest.fixture
def config():
    """Create a fast test configuration."""
    return LDDMMConfig(
        n_steps=2,  # Fewer steps for faster tests
        scale=5.0,  # Smaller scale for small synthetic data
        n_iter=5,   # Fewer iterations for faster tests
    )


# ============================================================================
# Tests for LDDMMConfig
# ============================================================================


class TestLDDMMConfig:
    """Tests for LDDMMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LDDMMConfig()
        assert config.n_steps == 5
        assert config.kernel == "gaussian"
        assert config.scale == 10.0
        assert config.regularization_weight == 0.01
        assert config.n_iter == 100

    def test_auto_device_selection(self):
        """Test that 'auto' device gets resolved."""
        config = LDDMMConfig(device="auto")
        assert config.device in ["cuda", "cpu"]

    def test_for_femurs_preset(self):
        """Test femur preset configuration."""
        config = LDDMMConfig.for_femurs()
        assert config.scale == 15.0
        assert config.n_steps == 5

    def test_high_precision_preset(self):
        """Test high precision preset."""
        config = LDDMMConfig.high_precision()
        assert config.n_steps == 10
        assert config.n_iter == 200

    def test_fast_preset(self):
        """Test fast preset configuration."""
        config = LDDMMConfig.fast()
        assert config.n_steps == 3
        assert config.n_iter == 50

    def test_invalid_n_steps_raises(self):
        """Test that invalid n_steps raises ValueError."""
        with pytest.raises(ValueError):
            LDDMMConfig(n_steps=0)

    def test_invalid_scale_raises(self):
        """Test that negative scale raises ValueError."""
        with pytest.raises(ValueError):
            LDDMMConfig(scale=-1.0)


# ============================================================================
# Tests for LDDMMRegistration
# ============================================================================


class TestLDDMMRegistrationInit:
    """Tests for LDDMMRegistration initialization."""

    def test_default_config(self):
        """Test initialization with default config."""
        try:
            reg = LDDMMRegistration()
            assert reg.config is not None
        except ImportError:
            pytest.skip("scikit-shapes not installed")

    def test_custom_config(self, config):
        """Test initialization with custom config."""
        try:
            reg = LDDMMRegistration(config)
            assert reg.config.n_steps == config.n_steps
        except ImportError:
            pytest.skip("scikit-shapes not installed")


class TestLDDMMRegistrationRegister:
    """Tests for LDDMM registration."""

    def test_register_returns_result(self, source_target, config):
        """Test that register returns a RegistrationResult."""
        try:
            source, target = source_target
            reg = LDDMMRegistration(config)
            result = reg.register(source, target)

            assert isinstance(result, RegistrationResult)
        except ImportError:
            pytest.skip("scikit-shapes not installed")

    def test_momentum_shape(self, source_target, config):
        """Test that momentum has correct shape."""
        try:
            source, target = source_target
            reg = LDDMMRegistration(config)
            result = reg.register(source, target)

            assert result.momentum.shape == source.shape
        except ImportError:
            pytest.skip("scikit-shapes not installed")

    def test_transformed_shape(self, source_target, config):
        """Test that transformed shape has correct dimensions."""
        try:
            source, target = source_target
            reg = LDDMMRegistration(config)
            result = reg.register(source, target)

            assert result.transformed.shape == target.shape
        except ImportError:
            pytest.skip("scikit-shapes not installed")

    def test_shape_mismatch_raises(self, config):
        """Test that mismatched shapes raise ValueError."""
        try:
            source = np.random.randn(100, 3).astype(np.float32)
            target = np.random.randn(50, 3).astype(np.float32)
            reg = LDDMMRegistration(config)

            with pytest.raises(ValueError, match="Shape mismatch"):
                reg.register(source, target)
        except ImportError:
            pytest.skip("scikit-shapes not installed")


class TestLDDMMRegistrationComputeMomentum:
    """Tests for compute_momentum method."""

    def test_compute_momentum_shape(self, source_target, config):
        """Test that compute_momentum returns correct shape."""
        try:
            source, target = source_target
            reg = LDDMMRegistration(config)
            momentum = reg.compute_momentum(source, target)

            assert momentum.shape == source.shape
        except ImportError:
            pytest.skip("scikit-shapes not installed")


class TestLDDMMRegistrationShoot:
    """Tests for shoot (exponential map) method."""

    def test_shoot_shape(self, source_target, config):
        """Test that shooting produces correct shape."""
        try:
            source, target = source_target
            reg = LDDMMRegistration(config)

            # Create a momentum
            momentum = np.random.randn(*source.shape).astype(np.float32) * 0.1

            result = reg.shoot(source, momentum)

            assert result.shape == source.shape
        except ImportError:
            pytest.skip("scikit-shapes not installed")

    def test_shoot_momentum_mismatch_raises(self, source_target, config):
        """Test that mismatched momentum raises ValueError."""
        try:
            source, _ = source_target
            reg = LDDMMRegistration(config)

            momentum = np.random.randn(50, 3).astype(np.float32)

            with pytest.raises(ValueError, match="Shape mismatch"):
                reg.shoot(source, momentum)
        except ImportError:
            pytest.skip("scikit-shapes not installed")


# ============================================================================
# Tests for RegistrationResult
# ============================================================================


class TestRegistrationResult:
    """Tests for RegistrationResult dataclass."""

    def test_creation(self):
        """Test creating a RegistrationResult."""
        momentum = np.random.randn(100, 3)
        transformed = np.random.randn(100, 3)

        result = RegistrationResult(
            momentum=momentum,
            transformed=transformed,
            path=None,
            energy=0.5,
            success=True,
        )

        assert result.momentum.shape == (100, 3)
        assert result.transformed.shape == (100, 3)
        assert result.energy == 0.5
        assert result.success is True
        assert result.path is None
