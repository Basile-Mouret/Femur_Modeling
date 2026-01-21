#!/usr/bin/env python3
"""
PCA Reconstruction Comparison

Compares reconstruction quality between Linear PCA and Tangent PCA (LDDMM-based).
Shows side-by-side heatmaps and summary metrics.

Usage:
    python pca_reconstruction_comparison.py <original_femur.obj> [--n_components 5]
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import numpy as np
import pyvista as pv
import trimesh
from typing import Tuple, Dict, Optional

from lddmm import TangentPCA, FemurDataLoader


def load_obj_vertices(obj_path: str) -> np.ndarray:
    """Load vertices from OBJ file."""
    mesh = trimesh.load(obj_path, process=False)
    return np.array(mesh.vertices, dtype=np.float32)


def load_obj_mesh(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from OBJ file."""
    mesh = trimesh.load(obj_path, process=False)
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces)


class LinearPCAModel:
    """Simple Linear PCA model for comparison."""
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ratio = None
        
    def fit(self, shapes: np.ndarray):
        """
        Fit Linear PCA on shapes.
        
        Args:
            shapes: Array of shape (n_samples, n_points, 3) or list of (n_points, 3)
        """
        if isinstance(shapes, list):
            shapes = np.stack(shapes, axis=0)
        n_samples = shapes.shape[0]
        # Flatten to (n_samples, n_points * 3)
        X = shapes.reshape(n_samples, -1)
        
        # Compute mean
        self.mean = X.mean(axis=0)
        
        # Center data
        X_centered = X - self.mean
        
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Store components
        n_comp = min(self.n_components, Vt.shape[0])
        self.components = Vt[:n_comp]  # (n_components, n_points * 3)
        
        # Explained variance
        total_var = (S ** 2).sum()
        self.explained_variance_ratio = (S[:n_comp] ** 2) / total_var
        
        # Store for projection
        self._singular_values = S[:n_comp]
        
    def project(self, shape: np.ndarray) -> np.ndarray:
        """Project a shape to the latent space."""
        x = shape.flatten() - self.mean
        return self.components @ x
    
    def reconstruct(self, shape: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """Reconstruct a shape using n_components."""
        if n_components is None:
            n_components = self.n_components
        n_components = min(n_components, len(self.components))
        
        # Project
        coeffs = self.project(shape)[:n_components]
        
        # Reconstruct
        reconstructed = self.mean + coeffs @ self.components[:n_components]
        return reconstructed.reshape(-1, 3)
    
    def save(self, path: str):
        """Save model to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "mean.npy", self.mean)
        np.save(path / "components.npy", self.components)
        np.save(path / "explained_variance_ratio.npy", self.explained_variance_ratio)
        
    @classmethod
    def load(cls, path: str) -> "LinearPCAModel":
        """Load model from directory."""
        path = Path(path)
        model = cls()
        model.mean = np.load(path / "mean.npy")
        model.components = np.load(path / "components.npy")
        model.explained_variance_ratio = np.load(path / "explained_variance_ratio.npy")
        model.n_components = len(model.components)
        return model


def compute_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute various reconstruction error metrics.
    
    Returns:
        Dictionary with metrics: rmse, max_error, mean_error, hausdorff
    """
    # Per-vertex L2 distances
    distances = np.linalg.norm(original - reconstructed, axis=1)
    
    return {
        "rmse": np.sqrt(np.mean(distances ** 2)),
        "mean_error": np.mean(distances),
        "max_error": np.max(distances),
        "median_error": np.median(distances),
        "std_error": np.std(distances),
        # Hausdorff is just max for corresponding points
        "hausdorff": np.max(distances),
    }


def create_error_mesh(vertices: np.ndarray, faces: np.ndarray, errors: np.ndarray) -> pv.PolyData:
    """Create PyVista mesh with error scalars."""
    pv_faces = np.hstack([[3] + list(f) for f in faces])
    mesh = pv.PolyData(vertices, pv_faces)
    mesh["Error (mm)"] = errors
    return mesh


def compare_reconstructions(
    original_path: str,
    linear_pca_model: LinearPCAModel,
    tangent_pca_model: TangentPCA,
    template_faces: np.ndarray,
    n_components_list: list = [1, 3, 5, 10],
):
    """
    Compare Linear PCA vs Tangent PCA reconstructions.
    
    Shows side-by-side heatmaps and prints metrics.
    """
    # Load original
    original = load_obj_vertices(original_path)
    
    print(f"\n{'='*70}")
    print(f"Reconstruction Comparison: {Path(original_path).name}")
    print(f"{'='*70}")
    
    results = {"linear": {}, "tangent": {}}
    
    for n_comp in n_components_list:
        if n_comp > linear_pca_model.n_components:
            continue
            
        print(f"\n--- {n_comp} Components ---")
        
        # Linear PCA reconstruction
        linear_recon = linear_pca_model.reconstruct(original, n_components=n_comp)
        linear_metrics = compute_reconstruction_metrics(original, linear_recon)
        results["linear"][n_comp] = linear_metrics
        
        # Tangent PCA reconstruction
        tangent_recon = tangent_pca_model.reconstruct(original, n_components=n_comp)
        tangent_metrics = compute_reconstruction_metrics(original, tangent_recon)
        results["tangent"][n_comp] = tangent_metrics
        
        print(f"  {'Metric':<15} {'Linear PCA':>12} {'Tangent PCA':>12} {'Diff':>10}")
        print(f"  {'-'*50}")
        for metric in ["rmse", "mean_error", "max_error", "median_error"]:
            lin_val = linear_metrics[metric]
            tan_val = tangent_metrics[metric]
            diff = tan_val - lin_val
            diff_str = f"{diff:+.4f}" if abs(diff) > 0.0001 else "~0"
            print(f"  {metric:<15} {lin_val:>12.4f} {tan_val:>12.4f} {diff_str:>10}")
    
    # Visualize best reconstruction (highest n_components)
    best_n = max(n_components_list)
    if best_n > linear_pca_model.n_components:
        best_n = linear_pca_model.n_components
        
    linear_recon = linear_pca_model.reconstruct(original, n_components=best_n)
    tangent_recon = tangent_pca_model.reconstruct(original, n_components=best_n)
    
    linear_errors = np.linalg.norm(original - linear_recon, axis=1)
    tangent_errors = np.linalg.norm(original - tangent_recon, axis=1)
    
    # Compute common color scale
    max_error = max(linear_errors.max(), tangent_errors.max())
    
    # Create visualization
    plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600))
    
    # Original
    plotter.subplot(0, 0)
    plotter.add_text("Original", font_size=14)
    orig_mesh = create_error_mesh(original, template_faces, np.zeros(len(original)))
    plotter.add_mesh(orig_mesh, color="beige", smooth_shading=True)
    
    # Linear PCA
    plotter.subplot(0, 1)
    plotter.add_text(f"Linear PCA ({best_n} comp)\nRMSE: {results['linear'][best_n]['rmse']:.3f}mm", font_size=12)
    linear_mesh = create_error_mesh(linear_recon, template_faces, linear_errors)
    plotter.add_mesh(
        linear_mesh, 
        scalars="Error (mm)", 
        cmap="coolwarm",
        clim=[0, max_error],
        smooth_shading=True,
        scalar_bar_args={"title": "Error (mm)", "n_labels": 5}
    )
    
    # Tangent PCA
    plotter.subplot(0, 2)
    plotter.add_text(f"Tangent PCA ({best_n} comp)\nRMSE: {results['tangent'][best_n]['rmse']:.3f}mm", font_size=12)
    tangent_mesh = create_error_mesh(tangent_recon, template_faces, tangent_errors)
    plotter.add_mesh(
        tangent_mesh,
        scalars="Error (mm)",
        cmap="coolwarm", 
        clim=[0, max_error],
        smooth_shading=True,
        scalar_bar_args={"title": "Error (mm)", "n_labels": 5}
    )
    
    plotter.link_views()
    plotter.show()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Linear PCA vs Tangent PCA reconstruction")
    parser.add_argument("femur", help="Path to original femur OBJ file")
    parser.add_argument("--n_components", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Number of components to test")
    parser.add_argument("--linear_model", type=str, default=None,
                        help="Path to Linear PCA model (will train if not provided)")
    parser.add_argument("--tangent_model", type=str, 
                        default="scripts/pca/model/tangent_pca",
                        help="Path to Tangent PCA model")
    parser.add_argument("--data_dir", type=str, default="data/training",
                        help="Training data directory (for fitting Linear PCA)")
    parser.add_argument("--template", type=str,
                        default="data/training/L_Femur_11_DECIM.obj.FINAL.obj",
                        help="Template mesh for faces")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Load template for faces
    template_path = project_root / args.template
    _, template_faces = load_obj_mesh(str(template_path))
    
    # Load or train Linear PCA
    if args.linear_model and Path(args.linear_model).exists():
        print("Loading Linear PCA model...")
        linear_model = LinearPCAModel.load(args.linear_model)
    else:
        print("Training Linear PCA model...")
        loader = FemurDataLoader(str(project_root / args.data_dir))
        shapes, _ = loader.load_all()
        linear_model = LinearPCAModel(n_components=max(args.n_components))
        linear_model.fit(shapes)
        
        # Save for future use
        save_path = project_root / "scripts/pca/model/linear_pca"
        linear_model.save(str(save_path))
        print(f"Saved Linear PCA model to {save_path}")
    
    # Load Tangent PCA
    print("Loading Tangent PCA model...")
    tangent_model = TangentPCA.load(str(project_root / args.tangent_model))
    
    # Run comparison
    femur_path = args.femur if Path(args.femur).is_absolute() else str(project_root / args.femur)
    compare_reconstructions(
        femur_path,
        linear_model,
        tangent_model,
        template_faces,
        n_components_list=args.n_components
    )


if __name__ == "__main__":
    main()
