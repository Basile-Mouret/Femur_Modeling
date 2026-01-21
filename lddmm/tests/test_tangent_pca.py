#!/usr/bin/env python3
"""Tests for LDDMM Tangent PCA module."""

import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from lddmm.tangent_pca import TangentPCA
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
def atlas_and_momenta(shapes_subset):
    """Build atlas and get momenta."""
    builder = LDDMMAtlasBuilder(max_outer_iterations=2, verbose=False)
    builder.build(shapes_subset)
    return builder.atlas, builder.momenta


@pytest.fixture
def pca():
    """Create an unfitted TangentPCA instance."""
    return TangentPCA(n_components=3)


@pytest.fixture
def fitted_pca(atlas_and_momenta):
    """Return a fitted TangentPCA instance."""
    atlas, momenta = atlas_and_momenta
    pca = TangentPCA(n_components=3)
    pca.fit(atlas, momenta)
    return pca


# ============================================================================
# Tests
# ============================================================================

class TestTangentPCAInit:
    """Tests for TangentPCA initialization."""
    
    def test_default_components(self):
        """Test default number of components is None (determined at fit time)."""
        pca = TangentPCA()
        assert pca.n_components is None
    
    def test_custom_components(self):
        """Test custom number of components."""
        pca = TangentPCA(n_components=5)
        assert pca.n_components == 5
    
    def test_unfitted_state(self, pca):
        """Test that new instance is unfitted."""
        assert pca.atlas is None
        assert pca.components is None
        assert pca.mean_momentum is None


class TestTangentPCAUnfittedErrors:
    """Tests for error handling on unfitted model."""
    
    def test_transform_raises_error(self, pca):
        """Test that transform raises error when not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.transform([np.zeros((100, 3))])
    
    def test_inverse_transform_raises_error(self, pca):
        """Test that inverse_transform raises error when not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.inverse_transform(np.zeros(3))
    
    def test_synthesize_shape_raises_error(self, pca):
        """Test that synthesize_shape raises error when not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.synthesize_shape(np.zeros(3))
    
    def test_synthesize_along_mode_raises_error(self, pca):
        """Test that synthesize_along_mode raises error when not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.synthesize_along_mode(0, np.array([-1, 0, 1]))


class TestTangentPCAFit:
    """Tests for TangentPCA fitting."""
    
    def test_fit_sets_atlas(self, fitted_pca, atlas_and_momenta):
        """Test that fit sets the atlas."""
        atlas, _ = atlas_and_momenta
        np.testing.assert_array_equal(fitted_pca.atlas, atlas)
    
    def test_fit_sets_components(self, fitted_pca):
        """Test that fit sets components."""
        assert fitted_pca.components is not None
        assert fitted_pca.components.shape[0] == fitted_pca.n_components
    
    def test_fit_sets_mean_momentum(self, fitted_pca, atlas_and_momenta):
        """Test that fit sets mean momentum."""
        _, momenta = atlas_and_momenta
        n_points = momenta[0].shape[0]
        
        assert fitted_pca.mean_momentum is not None
        assert fitted_pca.mean_momentum.shape == (n_points, 3)
    
    def test_explained_variance_ratio(self, fitted_pca):
        """Test that explained variance ratios are valid."""
        evr = fitted_pca.explained_variance_ratio
        
        assert len(evr) == fitted_pca.n_components
        assert np.all(evr >= 0)
        assert np.all(evr <= 1)
        assert np.sum(evr) <= 1.0 + 1e-6  # Allow small numerical error
    
    def test_explained_variance_decreasing(self, fitted_pca):
        """Test that explained variance is in decreasing order."""
        evr = fitted_pca.explained_variance_ratio
        for i in range(len(evr) - 1):
            assert evr[i] >= evr[i + 1] - 1e-6


class TestTangentPCATransform:
    """Tests for TangentPCA transform."""
    
    def test_transform_output_shape(self, fitted_pca, atlas_and_momenta):
        """Test transform output shape."""
        _, momenta = atlas_and_momenta
        coeffs = fitted_pca.transform(momenta)
        
        assert coeffs.shape == (len(momenta), fitted_pca.n_components)
    
    def test_transform_single_momentum(self, fitted_pca, atlas_and_momenta):
        """Test transform with single momentum."""
        _, momenta = atlas_and_momenta
        coeffs = fitted_pca.transform([momenta[0]])
        
        assert coeffs.shape == (1, fitted_pca.n_components)


class TestTangentPCAInverseTransform:
    """Tests for TangentPCA inverse transform."""
    
    def test_inverse_transform_single(self, fitted_pca):
        """Test inverse transform with single coefficient vector."""
        coeffs = np.zeros(fitted_pca.n_components)
        momentum = fitted_pca.inverse_transform(coeffs)
        
        # With zero coefficients, should get mean momentum
        np.testing.assert_array_almost_equal(momentum, fitted_pca.mean_momentum)
    
    def test_inverse_transform_batch(self, fitted_pca):
        """Test inverse transform with batch of coefficients."""
        coeffs = np.zeros((3, fitted_pca.n_components))
        momenta = fitted_pca.inverse_transform(coeffs)
        
        assert momenta.shape[0] == 3
    
    def test_roundtrip(self, fitted_pca, atlas_and_momenta):
        """Test transform -> inverse_transform roundtrip."""
        _, momenta = atlas_and_momenta
        
        coeffs = fitted_pca.transform(momenta)
        reconstructed = fitted_pca.inverse_transform(coeffs)
        
        # Reprojecting reconstructed momenta should give same coefficients
        reprojected_coeffs = fitted_pca.transform([reconstructed[i] for i in range(len(momenta))])
        
        # Use relative tolerance for float32 precision
        np.testing.assert_allclose(coeffs, reprojected_coeffs, rtol=1e-5)


class TestTangentPCASynthesize:
    """Tests for TangentPCA shape synthesis."""
    
    def test_synthesize_shape_output(self, fitted_pca, atlas_and_momenta):
        """Test synthesize_shape output shape."""
        atlas, _ = atlas_and_momenta
        coeffs = np.zeros(fitted_pca.n_components)
        shape = fitted_pca.synthesize_shape(coeffs)
        
        assert shape.shape == atlas.shape
    
    def test_synthesize_shape_at_mean(self, fitted_pca, atlas_and_momenta):
        """Test that zero coefficients give atlas + mean momentum."""
        coeffs = np.zeros(fitted_pca.n_components)
        shape = fitted_pca.synthesize_shape(coeffs)
        
        expected = fitted_pca.atlas + fitted_pca.mean_momentum
        # Use lower precision due to float32/float64 mixing
        np.testing.assert_array_almost_equal(shape, expected, decimal=4)
    
    def test_synthesize_along_mode_output(self, fitted_pca, atlas_and_momenta):
        """Test synthesize_along_mode output shape."""
        t_values = np.array([-2, -1, 0, 1, 2])
        shapes = fitted_pca.synthesize_along_mode(0, t_values)
        
        assert shapes.shape[0] == len(t_values)
    
    def test_synthesize_along_mode_invalid_mode(self, fitted_pca):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            fitted_pca.synthesize_along_mode(100, np.array([0]))
    
    def test_synthesize_along_mode_negative_index(self, fitted_pca):
        """Test that negative mode index raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            fitted_pca.synthesize_along_mode(-1, np.array([0]))


class TestTangentPCASaveLoad:
    """Tests for TangentPCA save/load functionality."""
    
    def test_save_creates_files(self, fitted_pca):
        """Test that save creates expected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_pca.save(tmpdir)
            
            tmppath = Path(tmpdir)
            assert (tmppath / "tangent_pca_atlas.npy").exists()
            assert (tmppath / "tangent_pca_components.npy").exists()
            assert (tmppath / "tangent_pca_metadata.json").exists()
    
    def test_load_restores_model(self, fitted_pca):
        """Test that load restores model correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_pca.save(tmpdir)
            loaded = TangentPCA.load(tmpdir)
            
            np.testing.assert_array_equal(loaded.atlas, fitted_pca.atlas)
            np.testing.assert_array_equal(loaded.components, fitted_pca.components)
            assert loaded.n_components == fitted_pca.n_components
    
    def test_loaded_model_can_synthesize(self, fitted_pca):
        """Test that loaded model can synthesize shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_pca.save(tmpdir)
            loaded = TangentPCA.load(tmpdir)
            
            coeffs = np.zeros(loaded.n_components)
            shape = loaded.synthesize_shape(coeffs)
            
            assert shape.shape == loaded.atlas.shape


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
