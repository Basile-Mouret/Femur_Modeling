#!/usr/bin/env python3
"""
Tests for Tangent PCA Visualization Module

Tests the visualizer and explorer components.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pca.tangent_pca_visualizer import (
    TangentPCAModel,
    load_tangent_pca_model,
    load_template_mesh,
    create_mesh_from_points,
    generate_tangent_shape,
    generate_tangent_mode_variation,
    TangentPCAVisualizer,
)
from scripts.pca.tangent_pca_explorer import TangentPCAExplorer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model() -> TangentPCAModel:
    """Create a mock Tangent PCA model for testing."""
    n_points = 100
    n_components = 5
    n_samples = 20
    
    # Create random but consistent data
    np.random.seed(42)
    
    atlas = np.random.randn(n_points, 3).astype(np.float32)
    mean_momentum = np.random.randn(n_points, 3).astype(np.float32) * 0.1
    components = np.random.randn(n_components, n_points, 3).astype(np.float32) * 0.05
    
    # Create decreasing variances
    variances = np.array([10.0, 5.0, 2.0, 1.0, 0.5], dtype=np.float32)
    total_variance = variances.sum()
    explained_variance_ratio = variances / total_variance
    
    return TangentPCAModel(
        atlas=atlas,
        mean_momentum=mean_momentum,
        components=components,
        variances=variances,
        n_dimensions=n_points * 3,
        n_samples=n_samples,
        n_components=n_components,
        total_variance=total_variance,
        explained_variance_ratio=explained_variance_ratio
    )


@pytest.fixture
def mock_template(mock_model: TangentPCAModel) -> pv.PolyData:
    """Create a mock template mesh (simple point cloud with faces)."""
    points = mock_model.atlas
    
    # Create a simple mesh with triangular faces
    n_points = len(points)
    
    # Use pyvista's sphere as a base and replace points
    sphere = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
    
    # Resample to match our point count if needed
    if sphere.n_points != n_points:
        # Create a simple triangulated mesh
        cloud = pv.PolyData(points)
        template = cloud.delaunay_2d()
        if template.n_cells == 0:
            # Fallback: create synthetic faces
            faces = []
            for i in range(0, n_points - 2, 3):
                faces.extend([3, i, i+1, i+2])
            if len(faces) > 0:
                template = pv.PolyData(points, faces=faces)
            else:
                template = pv.PolyData(points)
        return template
    
    sphere.points = points
    return sphere


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


# =============================================================================
# TangentPCAModel Tests
# =============================================================================

class TestTangentPCAModel:
    """Tests for the TangentPCAModel wrapper."""
    
    def test_model_attributes(self, mock_model):
        """Test that model has all required attributes."""
        assert hasattr(mock_model, 'atlas')
        assert hasattr(mock_model, 'mean_momentum')
        assert hasattr(mock_model, 'components')
        assert hasattr(mock_model, 'variances')
        assert hasattr(mock_model, 'explained_variance_ratio')
        assert hasattr(mock_model, 'n_components')
        assert hasattr(mock_model, 'n_samples')
    
    def test_model_shapes(self, mock_model):
        """Test that model arrays have correct shapes."""
        assert mock_model.atlas.shape[1] == 3
        assert mock_model.mean_momentum.shape == mock_model.atlas.shape
        assert mock_model.components.shape[0] == mock_model.n_components
        assert mock_model.components.shape[1:] == mock_model.atlas.shape
        assert len(mock_model.variances) == mock_model.n_components
        assert len(mock_model.explained_variance_ratio) == mock_model.n_components
    
    def test_variance_ratio_sums_to_one(self, mock_model):
        """Test that explained variance ratios sum to 1."""
        np.testing.assert_allclose(
            mock_model.explained_variance_ratio.sum(),
            1.0,
            rtol=1e-5
        )


# =============================================================================
# Shape Generation Tests
# =============================================================================

class TestShapeGeneration:
    """Tests for shape generation functions."""
    
    def test_generate_mean_shape(self, mock_model):
        """Test generating mean shape (all weights zero)."""
        weights = np.zeros(mock_model.n_components)
        shape = generate_tangent_shape(mock_model, weights)
        
        expected = mock_model.atlas + mock_model.mean_momentum
        np.testing.assert_allclose(shape, expected, rtol=1e-5)
    
    def test_generate_shape_with_weights(self, mock_model):
        """Test generating shape with non-zero weights."""
        weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        shape = generate_tangent_shape(mock_model, weights)
        
        # Shape should differ from mean
        mean_shape = mock_model.atlas + mock_model.mean_momentum
        assert not np.allclose(shape, mean_shape)
    
    def test_generate_mode_variation_mean(self, mock_model):
        """Test mode variation at sigma=0 gives mean."""
        shape = generate_tangent_mode_variation(mock_model, mode=0, sigma=0)
        expected = mock_model.atlas + mock_model.mean_momentum
        np.testing.assert_allclose(shape, expected, rtol=1e-5)
    
    def test_generate_mode_variation_positive(self, mock_model):
        """Test mode variation at positive sigma."""
        shape = generate_tangent_mode_variation(mock_model, mode=0, sigma=2.0)
        mean_shape = mock_model.atlas + mock_model.mean_momentum
        
        # Should be different from mean
        assert not np.allclose(shape, mean_shape)
    
    def test_generate_mode_variation_negative(self, mock_model):
        """Test mode variation at negative sigma."""
        shape_pos = generate_tangent_mode_variation(mock_model, mode=0, sigma=2.0)
        shape_neg = generate_tangent_mode_variation(mock_model, mode=0, sigma=-2.0)
        
        # Positive and negative should be symmetric around mean
        mean_shape = mock_model.atlas + mock_model.mean_momentum
        diff_pos = shape_pos - mean_shape
        diff_neg = shape_neg - mean_shape
        
        # Use slightly looser tolerance for float32 precision
        np.testing.assert_allclose(diff_pos, -diff_neg, rtol=1e-4)
    
    def test_mode_out_of_range(self, mock_model):
        """Test that out-of-range mode raises error."""
        with pytest.raises(ValueError):
            generate_tangent_mode_variation(mock_model, mode=100, sigma=1.0)


# =============================================================================
# Mesh Utilities Tests
# =============================================================================

class TestMeshUtilities:
    """Tests for mesh creation utilities."""
    
    def test_create_mesh_from_points(self, mock_model, mock_template):
        """Test creating mesh from points."""
        new_points = mock_model.atlas + np.random.randn(*mock_model.atlas.shape) * 0.1
        mesh = create_mesh_from_points(new_points, mock_template)
        
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points == mock_template.n_points
        np.testing.assert_allclose(mesh.points, new_points, rtol=1e-5)
    
    def test_template_not_found(self):
        """Test that missing template raises error."""
        with pytest.raises(FileNotFoundError):
            load_template_mesh('/nonexistent/path/template.obj')


# =============================================================================
# Visualizer Tests
# =============================================================================

class TestTangentPCAVisualizer:
    """Tests for TangentPCAVisualizer class."""
    
    def test_visualizer_initialization(self, mock_model, mock_template):
        """Test visualizer initialization."""
        viz = TangentPCAVisualizer(mock_model, mock_template)
        
        assert viz.n_vertices == mock_model.atlas.shape[0]
        assert len(viz.variance_ratios) == mock_model.n_components
    
    def test_dimension_mismatch_error(self, mock_model):
        """Test that mismatched dimensions raise error."""
        # Create a template with different vertex count
        wrong_template = pv.Sphere(radius=1.0, theta_resolution=5, phi_resolution=5)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            TangentPCAVisualizer(mock_model, wrong_template)
    
    def test_create_mesh_method(self, mock_model, mock_template):
        """Test internal mesh creation."""
        viz = TangentPCAVisualizer(mock_model, mock_template)
        
        points = mock_model.atlas + mock_model.mean_momentum
        mesh = viz._create_mesh(points)
        
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points == mock_model.atlas.shape[0]
    
    def test_cumulative_variance(self, mock_model, mock_template):
        """Test cumulative variance computation."""
        viz = TangentPCAVisualizer(mock_model, mock_template)
        
        # Cumulative should be monotonically increasing
        assert all(np.diff(viz.cumulative_variance) >= 0)
        
        # Should end at 1.0
        np.testing.assert_allclose(viz.cumulative_variance[-1], 1.0, rtol=1e-5)


# =============================================================================
# Explorer Tests
# =============================================================================

class TestTangentPCAExplorer:
    """Tests for TangentPCAExplorer class (without GUI)."""
    
    def test_explorer_weight_initialization(self, mock_model, mock_template, tmp_path):
        """Test that explorer initializes with zero weights."""
        # Save mock model to temp directory using TangentPCA file naming convention
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        
        # Use the same file naming as TangentPCA.save()
        np.save(model_dir / "tangent_pca_atlas.npy", mock_model.atlas)
        np.save(model_dir / "tangent_pca_mean_momentum.npy", mock_model.mean_momentum)
        np.save(model_dir / "tangent_pca_components.npy", mock_model.components)
        np.save(model_dir / "tangent_pca_eigenvalues.npy", mock_model.variances)
        np.save(model_dir / "tangent_pca_explained_variance.npy", 
                mock_model.explained_variance_ratio)
        
        metadata = {
            "n_samples": mock_model.n_samples,
            "n_components": mock_model.n_components,
            "n_points": mock_model.atlas.shape[0],
        }
        import json
        with open(model_dir / "tangent_pca_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Save template
        template_path = tmp_path / "template.obj"
        mock_template.save(str(template_path))
        
        # Create explorer (won't open GUI)
        explorer = TangentPCAExplorer(
            model_dir=str(model_dir),
            template_path=str(template_path),
            n_components=3
        )
        
        assert explorer.n_components == 3
        np.testing.assert_array_equal(explorer.weights, np.zeros(3))
    
    def test_controls_text_generation(self, mock_model, mock_template, tmp_path):
        """Test controls text generation."""
        # Save mock model using TangentPCA file naming convention
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        
        np.save(model_dir / "tangent_pca_atlas.npy", mock_model.atlas)
        np.save(model_dir / "tangent_pca_mean_momentum.npy", mock_model.mean_momentum)
        np.save(model_dir / "tangent_pca_components.npy", mock_model.components)
        np.save(model_dir / "tangent_pca_eigenvalues.npy", mock_model.variances)
        np.save(model_dir / "tangent_pca_explained_variance.npy",
                mock_model.explained_variance_ratio)
        
        metadata = {
            "n_samples": mock_model.n_samples,
            "n_components": mock_model.n_components,
            "n_points": mock_model.atlas.shape[0],
        }
        import json
        with open(model_dir / "tangent_pca_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        template_path = tmp_path / "template.obj"
        mock_template.save(str(template_path))
        
        explorer = TangentPCAExplorer(
            model_dir=str(model_dir),
            template_path=str(template_path),
            n_components=3
        )
        
        text = explorer._get_controls_text()
        
        assert "TANGENT PCA EXPLORER" in text
        assert "PC1:" in text
        assert "Variance Explained:" in text


# =============================================================================
# Integration Tests (with actual data if available)
# =============================================================================

class TestIntegration:
    """Integration tests using actual data (if available)."""
    
    def test_load_saved_model(self, project_root):
        """Test loading a saved Tangent PCA model."""
        model_dir = project_root / "visualization" / "model" / "tangent_pca"
        
        if not model_dir.exists():
            pytest.skip("No saved Tangent PCA model found")
        
        model = load_tangent_pca_model(str(model_dir))
        
        assert model.atlas is not None
        assert model.components is not None
        assert model.n_components > 0
    
    def test_load_actual_template(self, project_root):
        """Test loading an actual template mesh."""
        data_dir = project_root / "data" / "training"
        
        if not data_dir.exists():
            pytest.skip("No training data found")
        
        obj_files = list(data_dir.glob("*.obj"))
        if not obj_files:
            pytest.skip("No OBJ files found")
        
        template = load_template_mesh(str(obj_files[0]))
        
        assert isinstance(template, pv.PolyData)
        assert template.n_points > 0
        assert template.n_cells > 0


# =============================================================================
# Export Tests
# =============================================================================

class TestExport:
    """Tests for export functionality."""
    
    def test_export_atlas(self, mock_model, mock_template, tmp_path):
        """Test exporting atlas shape."""
        viz = TangentPCAVisualizer(mock_model, mock_template)
        
        output_path = str(tmp_path / "atlas.obj")
        viz.export_atlas(output_path)
        
        assert os.path.exists(output_path)
        
        # Verify it can be loaded
        loaded = pv.read(output_path)
        assert loaded.n_points == mock_model.atlas.shape[0]
    
    def test_export_mode_variations(self, mock_model, mock_template, tmp_path):
        """Test exporting mode variations."""
        viz = TangentPCAVisualizer(mock_model, mock_template)
        
        output_dir = str(tmp_path / "variations")
        viz.export_mode_variations(output_dir, n_modes=2, sigmas=[-1, 0, 1])
        
        assert os.path.exists(output_dir)
        
        # Check files were created
        files = os.listdir(output_dir)
        assert 'atlas_shape.obj' in files
        assert any('PC1' in f for f in files)
        assert any('PC2' in f for f in files)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
