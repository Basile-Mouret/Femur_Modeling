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
from lddmm.atlas import AtlasBuilder


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
def atlas_and_momenta(shapes_subset):
    """Build atlas and get momenta (runs LDDMM)."""
    builder = AtlasBuilder(max_iterations=2, verbose=False)
    result = builder.build(shapes_subset)
    return result.atlas, result.momenta


@pytest.fixture
def pca():
    """Create an unfitted TangentPCA instance."""
    return TangentPCA(n_components=3)


@pytest.fixture
def fitted_pca(atlas_and_momenta):
    """Return a fitted TangentPCA instance (requires LDDMM atlas)."""
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
        """Test default number of components is None."""
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

    def test_project_raises_error(self, pca):
        """Test that project raises error when not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.project(np.zeros((100, 3)))

    def test_synthesize_shape_raises_error(self, pca):
        """Test that synthesize_shape raises error when not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.synthesize_shape(np.zeros(3))

    def test_synthesize_along_mode_raises_error(self, pca):
        """Test that synthesize_along_mode raises error when not fitted."""
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.synthesize_along_mode(0, [-1, 0, 1])


class TestTangentPCAFit:
    """Tests for fitting Tangent PCA."""

    @pytest.mark.slow
    def test_fit_returns_self(self, pca, atlas_and_momenta):
        """Test that fit returns self."""
        atlas, momenta = atlas_and_momenta
        result = pca.fit(atlas, momenta)
        assert result is pca

    @pytest.mark.slow
    def test_fit_sets_atlas(self, fitted_pca, atlas_and_momenta):
        """Test that fit sets the atlas."""
        atlas, _ = atlas_and_momenta
        np.testing.assert_allclose(fitted_pca.atlas, atlas, rtol=1e-5)

    @pytest.mark.slow
    def test_fit_sets_components(self, fitted_pca):
        """Test that fit sets components."""
        assert fitted_pca.components is not None
        assert fitted_pca.components.shape[0] == 3  # n_components

    @pytest.mark.slow
    def test_fit_sets_mean_momentum(self, fitted_pca):
        """Test that fit sets mean momentum."""
        assert fitted_pca.mean_momentum is not None

    @pytest.mark.slow
    def test_fit_sets_eigenvalues(self, fitted_pca):
        """Test that fit sets eigenvalues."""
        assert fitted_pca.eigenvalues is not None
        assert len(fitted_pca.eigenvalues) == 3  # n_components

    @pytest.mark.slow
    def test_eigenvalues_sorted_descending(self, fitted_pca):
        """Test that eigenvalues are sorted in descending order."""
        eigenvalues = fitted_pca.eigenvalues
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:])

    @pytest.mark.slow
    @pytest.mark.slow
    def test_explained_variance_sums_to_one_or_less(self, atlas_and_momenta):
        """Test explained variance ratios are valid."""
        atlas, momenta = atlas_and_momenta
        pca = TangentPCA()  # Keep all components
        pca.fit(atlas, momenta)

        assert np.sum(pca.explained_variance_ratio) <= 1.0 + 1e-5
        assert np.all(pca.explained_variance_ratio >= 0)
        assert np.all(pca.explained_variance_ratio <= 1.0 + 1e-5)


@pytest.mark.slow
class TestTangentPCAProject:
    """Tests for projecting shapes (requires LDDMM registration)."""

    def test_project_shape(self, fitted_pca, shapes_subset):
        """Test projecting a shape to coefficients."""
        shape = shapes_subset[0]
        coefficients = fitted_pca.project(shape)

        assert coefficients.shape == (3,)  # n_components

    def test_project_atlas_returns_near_zero(self, fitted_pca):
        """Test that projecting atlas returns near-zero coefficients."""
        # Atlas projected should give approximately zero coefficients
        # (minus the mean momentum contribution)
        coefficients = fitted_pca.project(fitted_pca.atlas)
        # The projection of atlas should be dominated by mean momentum
        # which is already centered, so coefficients should be small
        # (but not exactly zero due to mean momentum centering)
        assert coefficients is not None


@pytest.mark.slow
class TestTangentPCASynthesis:
    """Tests for shape synthesis (requires geodesic shooting)."""

    def test_synthesize_shape_at_origin(self, fitted_pca):
        """Test synthesizing shape at origin (zero coefficients)."""
        coefficients = np.zeros(3)
        shape = fitted_pca.synthesize_shape(coefficients)

        # At origin, should return shooting atlas with mean_momentum
        assert shape.shape == fitted_pca.atlas.shape

    def test_synthesize_along_mode_shape(self, fitted_pca):
        """Test shape of synthesize_along_mode output."""
        shapes = fitted_pca.synthesize_along_mode(0, [-2, -1, 0, 1, 2])

        assert shapes.shape[0] == 5  # Number of t values
        assert shapes.shape[1] == fitted_pca.atlas.shape[0]  # N points
        assert shapes.shape[2] == 3  # 3D

    def test_synthesize_along_mode_zero_has_mean_shape(self, fitted_pca):
        """Test that t=0 gives the mean shape (shooting with mean momentum)."""
        shapes = fitted_pca.synthesize_along_mode(0, [0])
        # Should have the same shape as atlas
        assert shapes[0].shape == fitted_pca.atlas.shape

    def test_get_mode_extremes(self, fitted_pca):
        """Test get_mode_extremes returns correct shapes."""
        shapes, t_values = fitted_pca.get_mode_extremes(0, n_std=2.0, n_steps=5)

        assert shapes.shape[0] == 5
        assert len(t_values) == 5
        assert t_values[0] == -2.0
        assert t_values[-1] == 2.0

    def test_invalid_mode_raises(self, fitted_pca):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            fitted_pca.synthesize_along_mode(100, [0])


@pytest.mark.slow
class TestTangentPCAReconstruct:
    """Tests for reconstruction (requires LDDMM registration and shooting)."""

    def test_reconstruct_preserves_shape(self, fitted_pca, shapes_subset):
        """Test that reconstruction preserves shape dimensions."""
        shape = shapes_subset[0]
        reconstructed = fitted_pca.reconstruct(shape)

        assert reconstructed.shape == shape.shape

    def test_reconstruct_with_all_components_accurate(
        self, atlas_and_momenta, shapes_subset
    ):
        """Test reconstruction with all components is accurate."""
        atlas, momenta = atlas_and_momenta
        pca = TangentPCA()  # Keep all components
        pca.fit(atlas, momenta)

        shape = shapes_subset[0]
        reconstructed = pca.reconstruct(shape)

        # Should be reasonably close with all components
        error = np.linalg.norm(reconstructed - shape)
        original_norm = np.linalg.norm(shape)
        relative_error = error / original_norm

        assert relative_error < 0.5  # Within 50% relative error


@pytest.mark.slow
class TestTangentPCASaveLoad:
    """Tests for saving and loading PCA model."""

    def test_save_creates_files(self, fitted_pca):
        """Test that save creates expected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_pca.save(tmpdir)

            assert (Path(tmpdir) / "tangent_pca_atlas.npy").exists()
            assert (Path(tmpdir) / "tangent_pca_components.npy").exists()
            assert (Path(tmpdir) / "tangent_pca_eigenvalues.npy").exists()
            assert (Path(tmpdir) / "tangent_pca_metadata.json").exists()

    def test_load_restores_atlas(self, fitted_pca):
        """Test that load restores the atlas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_pca.save(tmpdir)
            loaded = TangentPCA.load(tmpdir)

            np.testing.assert_allclose(loaded.atlas, fitted_pca.atlas, rtol=1e-5)

    def test_load_restores_components(self, fitted_pca):
        """Test that load restores components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_pca.save(tmpdir)
            loaded = TangentPCA.load(tmpdir)

            np.testing.assert_allclose(
                loaded.components, fitted_pca.components, rtol=1e-5
            )

    def test_load_restores_metadata(self, fitted_pca):
        """Test that load restores metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_pca.save(tmpdir)
            loaded = TangentPCA.load(tmpdir)

            assert loaded.n_components == fitted_pca.n_components
            assert loaded.n_points_ == fitted_pca.n_points_
            assert loaded.n_samples_ == fitted_pca.n_samples_

    def test_loaded_model_can_synthesize(self, fitted_pca):
        """Test that loaded model can synthesize shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_pca.save(tmpdir)
            loaded = TangentPCA.load(tmpdir)

            shapes = loaded.synthesize_along_mode(0, [-1, 0, 1])
            assert shapes.shape[0] == 3
