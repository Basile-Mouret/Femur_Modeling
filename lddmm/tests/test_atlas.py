#!/usr/bin/env python3
"""Tests for LDDMM atlas builder module."""

import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from lddmm.atlas import LDDMMAtlasBuilder
from lddmm.data_loader import FemurDataLoader


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def data_dir():
    """Return path to training data directory."""
    return Path(__file__).parent.parent.parent / "data" / "training"


@pytest.fixture
def shapes_subset(data_dir):
    """Load a small subset of shapes for fast tests."""
    loader = FemurDataLoader(str(data_dir))
    shapes, _ = loader.load_all()
    return shapes[:5]


@pytest.fixture
def builder():
    """Create an atlas builder instance."""
    return LDDMMAtlasBuilder(
        max_outer_iterations=2,
        verbose=False
    )


@pytest.fixture
def fitted_builder(builder, shapes_subset):
    """Return a fitted atlas builder."""
    builder.build(shapes_subset)
    return builder


# ============================================================================
# Tests
# ============================================================================

class TestLDDMMAtlasBuilderInit:
    """Tests for LDDMMAtlasBuilder initialization."""
    
    def test_default_params(self):
        """Test default initialization."""
        builder = LDDMMAtlasBuilder()
        assert builder.atlas_step_size == 0.5
        assert builder.max_outer_iterations == 10
        assert builder.convergence_tol == 1e-4
    
    def test_custom_params(self):
        """Test custom initialization."""
        builder = LDDMMAtlasBuilder(
            atlas_step_size=0.3,
            max_outer_iterations=5,
            convergence_tol=1e-5
        )
        assert builder.atlas_step_size == 0.3
        assert builder.max_outer_iterations == 5
        assert builder.convergence_tol == 1e-5


class TestLDDMMAtlasBuilderBuild:
    """Tests for atlas building."""
    
    def test_build_returns_atlas(self, builder, shapes_subset):
        """Test that build returns an atlas."""
        result = builder.build(shapes_subset)
        
        assert builder.atlas is not None
        assert isinstance(builder.atlas, np.ndarray)
    
    def test_atlas_shape(self, fitted_builder, shapes_subset):
        """Test that atlas has correct shape."""
        expected_shape = shapes_subset[0].shape
        assert fitted_builder.atlas.shape == expected_shape
    
    def test_momenta_count(self, fitted_builder, shapes_subset):
        """Test that momenta count matches shape count."""
        assert len(fitted_builder.momenta) == len(shapes_subset)
    
    def test_momenta_shapes(self, fitted_builder, shapes_subset):
        """Test that each momentum has correct shape."""
        expected_shape = shapes_subset[0].shape
        for m in fitted_builder.momenta:
            assert m.shape == expected_shape
    
    def test_energy_history_recorded(self, fitted_builder):
        """Test that energy history is recorded."""
        assert len(fitted_builder.energy_history) > 0
    
    def test_custom_initial_atlas(self, shapes_subset):
        """Test using a custom initial atlas."""
        initial = shapes_subset[0].copy()
        builder = LDDMMAtlasBuilder(max_outer_iterations=1, verbose=False)
        builder.build(shapes_subset, initial_atlas=initial)
        
        # Atlas should be updated from initial
        assert builder.atlas is not None


class TestLDDMMAtlasBuilderSaveLoad:
    """Tests for save/load functionality."""
    
    def test_save_creates_files(self, fitted_builder):
        """Test that save creates expected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_builder.save(tmpdir)
            
            tmppath = Path(tmpdir)
            assert (tmppath / "atlas.npy").exists()
            assert (tmppath / "momenta.npy").exists()
            assert (tmppath / "energy_history.json").exists()
    
    def test_load_restores_atlas(self, fitted_builder):
        """Test that load restores atlas correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_builder.save(tmpdir)
            loaded = LDDMMAtlasBuilder.load(tmpdir)
            
            np.testing.assert_array_equal(loaded.atlas, fitted_builder.atlas)
    
    def test_load_restores_momenta(self, fitted_builder):
        """Test that load restores momenta correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_builder.save(tmpdir)
            loaded = LDDMMAtlasBuilder.load(tmpdir)
            
            assert len(loaded.momenta) == len(fitted_builder.momenta)
            for m1, m2 in zip(loaded.momenta, fitted_builder.momenta):
                np.testing.assert_array_equal(m1, m2)


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
