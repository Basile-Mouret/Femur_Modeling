#!/usr/bin/env python3
"""Tests for LDDMM atlas builder module."""

import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from lddmm.atlas import AtlasBuilder, AtlasResult
from lddmm.config import LDDMMConfig


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def shapes_subset():
    """Create small synthetic shapes for fast testing."""
    np.random.seed(42)
    n_shapes = 5
    n_points = 100  # Small point cloud for fast tests
    
    shapes = []
    for i in range(n_shapes):
        # Create sphere-like shape with variation
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = 10.0 + i * 0.5  # Slightly different radii
        
        shape = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ]).astype(np.float32)
        
        # Add random variation
        shape += np.random.randn(n_points, 3).astype(np.float32) * 0.3
        shapes.append(shape)
    
    return shapes


@pytest.fixture
def builder():
    """Create an atlas builder instance with fast settings."""
    return AtlasBuilder(
        max_iterations=2,
        verbose=False,
    )


@pytest.fixture
def fitted_builder(builder, shapes_subset):
    """Return a fitted atlas builder (runs LDDMM, marked slow)."""
    builder.build(shapes_subset)
    return builder


# ============================================================================
# Tests
# ============================================================================


class TestAtlasBuilderInit:
    """Tests for AtlasBuilder initialization."""

    def test_default_params(self):
        """Test default initialization."""
        builder = AtlasBuilder()
        assert builder.step_size == 0.5
        assert builder.max_iterations == 10
        assert builder.convergence_tol == 1e-4

    def test_custom_params(self):
        """Test custom initialization."""
        builder = AtlasBuilder(
            step_size=0.3,
            max_iterations=5,
            convergence_tol=1e-5,
        )
        assert builder.step_size == 0.3
        assert builder.max_iterations == 5
        assert builder.convergence_tol == 1e-5


class TestAtlasBuilderBuild:
    """Tests for atlas building (uses true LDDMM geodesic averaging)."""

    @pytest.mark.slow
    def test_build_returns_result(self, builder, shapes_subset):
        """Test that build returns an AtlasResult."""
        result = builder.build(shapes_subset)
        assert isinstance(result, AtlasResult)

    @pytest.mark.slow
    def test_atlas_shape(self, fitted_builder, shapes_subset):
        """Test that atlas has correct shape."""
        expected_shape = shapes_subset[0].shape
        assert fitted_builder.atlas.shape == expected_shape

    @pytest.mark.slow
    def test_momenta_shape(self, fitted_builder, shapes_subset):
        """Test that momenta array has correct shape."""
        K = len(shapes_subset)
        N, D = shapes_subset[0].shape
        assert fitted_builder.momenta.shape == (K, N, D)

    @pytest.mark.slow
    def test_convergence_history(self, fitted_builder):
        """Test that convergence history is recorded."""
        assert len(fitted_builder.convergence_history) > 0


class TestAtlasBuilderSaveLoad:
    """Tests for saving and loading atlas."""

    @pytest.mark.slow
    def test_save_creates_files(self, fitted_builder):
        """Test that save creates expected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_builder.save(tmpdir)

            assert (Path(tmpdir) / "atlas.npy").exists()
            assert (Path(tmpdir) / "momenta.npy").exists()
            assert (Path(tmpdir) / "atlas_metadata.json").exists()

    @pytest.mark.slow
    def test_load_restores_atlas(self, fitted_builder):
        """Test that load restores the atlas correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_builder.save(tmpdir)
            loaded = AtlasBuilder.load(tmpdir)

            np.testing.assert_allclose(
                loaded.atlas, fitted_builder.atlas, rtol=1e-5
            )

    @pytest.mark.slow
    def test_load_restores_momenta(self, fitted_builder):
        """Test that load restores momenta correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_builder.save(tmpdir)
            loaded = AtlasBuilder.load(tmpdir)

            np.testing.assert_allclose(
                loaded.momenta, fitted_builder.momenta, rtol=1e-5
            )


class TestAtlasBuilderValidation:
    """Tests for input validation."""

    def test_too_few_shapes_raises(self, builder):
        """Test that fewer than 2 shapes raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 shapes"):
            builder.build([np.random.randn(100, 3)])

    def test_inconsistent_shapes_raises(self, builder):
        """Test that inconsistent shapes raise ValueError."""
        shapes = [
            np.random.randn(100, 3),
            np.random.randn(50, 3),  # Different size
        ]
        with pytest.raises(ValueError, match="dimensions"):
            builder.build(shapes)


# ============================================================================
# Tests for AtlasResult
# ============================================================================


class TestAtlasResult:
    """Tests for AtlasResult dataclass."""

    def test_creation(self):
        """Test creating an AtlasResult."""
        atlas = np.random.randn(100, 3)
        momenta = np.random.randn(5, 100, 3)

        result = AtlasResult(
            atlas=atlas,
            momenta=momenta,
            convergence_history=[100.0, 50.0, 25.0],
        )

        assert result.atlas.shape == (100, 3)
        assert result.momenta.shape == (5, 100, 3)
        assert len(result.convergence_history) == 3
