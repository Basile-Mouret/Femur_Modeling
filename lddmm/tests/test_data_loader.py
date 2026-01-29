#!/usr/bin/env python3
"""Tests for LDDMM data loader module."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from lddmm.femur_lddmm.data_loader import (
    FemurDataLoader,
    verify_correspondence,
    compute_bounding_box
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def data_dir():
    """Return path to training data directory."""
    return Path(__file__).parent.parent.parent / "data" / "training"


@pytest.fixture
def loader(data_dir):
    """Create a FemurDataLoader instance."""
    return FemurDataLoader(str(data_dir))


@pytest.fixture
def shapes(loader):
    """Load all shapes."""
    shapes, _ = loader.load_all()
    return shapes


# ============================================================================
# Tests
# ============================================================================

class TestFemurDataLoader:
    """Tests for FemurDataLoader class."""
    
    def test_loader_finds_files(self, loader):
        """Test that loader finds OBJ files."""
        assert len(loader) > 0, "No files found in data directory"
    
    def test_load_all_returns_shapes_and_filenames(self, loader):
        """Test that load_all returns correct types."""
        shapes, filenames = loader.load_all()
        
        assert isinstance(shapes, list)
        assert isinstance(filenames, list)
        assert len(shapes) == len(filenames)
        assert len(shapes) > 0
    
    def test_shapes_are_3d_arrays(self, shapes):
        """Test that all shapes are Nx3 numpy arrays."""
        for shape in shapes:
            assert isinstance(shape, np.ndarray)
            assert shape.ndim == 2
            assert shape.shape[1] == 3
    
    def test_shapes_have_same_vertex_count(self, shapes):
        """Test point correspondence (same vertex count)."""
        n_vertices = shapes[0].shape[0]
        for shape in shapes:
            assert shape.shape[0] == n_vertices


class TestVerifyCorrespondence:
    """Tests for verify_correspondence function."""
    
    def test_valid_shapes_pass(self, shapes):
        """Test that valid shapes pass verification."""
        assert verify_correspondence(shapes, verbose=False)
    
    def test_empty_list_fails(self):
        """Test that empty list fails verification."""
        assert not verify_correspondence([], verbose=False)
    
    def test_mismatched_vertex_count_fails(self):
        """Test that shapes with different vertex counts fail."""
        shapes = [
            np.random.rand(100, 3),
            np.random.rand(50, 3)  # Different count
        ]
        assert not verify_correspondence(shapes, verbose=False)
    
    def test_wrong_dimensions_fails(self):
        """Test that shapes with wrong dimensions fail."""
        shapes = [
            np.random.rand(100, 3),
            np.random.rand(100, 2)  # 2D instead of 3D
        ]
        assert not verify_correspondence(shapes, verbose=False)


class TestComputeBoundingBox:
    """Tests for compute_bounding_box function."""
    
    def test_bounding_box_keys(self, shapes):
        """Test that bounding box has expected keys."""
        bbox = compute_bounding_box(shapes)
        
        assert 'min' in bbox
        assert 'max' in bbox
        assert 'center' in bbox
        assert 'size' in bbox
    
    def test_bounding_box_values_shape(self, shapes):
        """Test that bounding box values are 3D vectors."""
        bbox = compute_bounding_box(shapes)
        
        assert bbox['min'].shape == (3,)
        assert bbox['max'].shape == (3,)
        assert bbox['center'].shape == (3,)
        assert bbox['size'].shape == (3,)
    
    def test_bounding_box_consistency(self, shapes):
        """Test that min <= center <= max."""
        bbox = compute_bounding_box(shapes)
        
        assert np.all(bbox['min'] <= bbox['center'])
        assert np.all(bbox['center'] <= bbox['max'])
        assert np.all(bbox['size'] >= 0)


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
