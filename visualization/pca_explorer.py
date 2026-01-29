#!/usr/bin/env python3
"""
Interactive PCA Shape Explorer with Deformation Grid.

Explore shape variation modes via sliders. Supports:
- Linear PCA models (mean + components)
- Tangent PCA / LDDMM models (atlas + momenta + components)

Usage:
    python -m visualization.pca_explorer --model models/lddmm_pca --template data/training/L_Femur_11_DECIM.obj.FINAL.obj

Controls:
    Sliders: Adjust PC weights (in standard deviations)
    G: Toggle deformation grid
    H: Toggle heatmap mode (deviation from mean)
    R: Reset all to mean shape
    Q: Quit
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyvista as pv


# =============================================================================
# Model Loading
# =============================================================================

@dataclass
class PCAModel:
    """Unified interface for Linear and Tangent PCA models."""

    mean: np.ndarray  # Mean shape (N, 3)
    components: np.ndarray  # Principal components (K, N, 3)
    variances: np.ndarray  # Eigenvalues (K,)
    explained_variance_ratio: np.ndarray  # Fraction per component (K,)
    n_components: int
    n_points: int
    model_type: str  # "linear" or "tangent"


def load_model(model_dir: str) -> PCAModel:
    """Load a PCA model from directory (auto-detects type)."""
    model_path = Path(model_dir)

    tangent_atlas = model_path / "tangent_pca_atlas.npy"
    tangent_components = model_path / "tangent_pca_components.npy"
    linear_mean = model_path / "mean.npy"
    linear_components = model_path / "components.npy"

    if tangent_atlas.exists() and tangent_components.exists():
        return _load_tangent_model(model_path)
    elif linear_mean.exists() and linear_components.exists():
        return _load_linear_model(model_path)
    else:
        raise FileNotFoundError(
            f"No valid PCA model found in {model_dir}. "
            "Expected tangent_pca_*.npy or mean.npy + components.npy"
        )


def _load_tangent_model(model_path: Path) -> PCAModel:
    """Load Tangent PCA model."""
    atlas = np.load(model_path / "tangent_pca_atlas.npy")
    mean_momentum = np.load(model_path / "tangent_pca_mean_momentum.npy")
    components = np.load(model_path / "tangent_pca_components.npy")
    eigenvalues = np.load(model_path / "tangent_pca_eigenvalues.npy")
    explained_var = np.load(model_path / "tangent_pca_explained_variance.npy")

    mean = atlas + mean_momentum

    print(f"[PCAExplorer] Loaded Tangent PCA: {components.shape[0]} components, {atlas.shape[0]} points")

    return PCAModel(
        mean=mean,
        components=components,
        variances=eigenvalues,
        explained_variance_ratio=explained_var,
        n_components=components.shape[0],
        n_points=atlas.shape[0],
        model_type="tangent",
    )


def _load_linear_model(model_path: Path) -> PCAModel:
    """Load Linear PCA model."""
    mean_flat = np.load(model_path / "mean.npy")
    components_flat = np.load(model_path / "components.npy")
    explained_var = np.load(model_path / "explained_variance_ratio.npy")

    n_points = len(mean_flat) // 3
    mean = mean_flat.reshape(n_points, 3)
    components = components_flat.reshape(-1, n_points, 3)

    total_var = np.sum((components_flat**2).sum(axis=1))
    variances = explained_var * total_var

    print(f"[PCAExplorer] Loaded Linear PCA: {components.shape[0]} components, {n_points} points")

    return PCAModel(
        mean=mean,
        components=components,
        variances=variances,
        explained_variance_ratio=explained_var,
        n_components=components.shape[0],
        n_points=n_points,
        model_type="linear",
    )


def generate_shape(model: PCAModel, weights: np.ndarray) -> np.ndarray:
    """Generate shape from PCA weights (in standard deviations)."""
    shape = model.mean.copy()
    n_weights = min(len(weights), model.n_components)

    for k in range(n_weights):
        std = np.sqrt(model.variances[k])
        shape = shape + weights[k] * std * model.components[k]

    return shape


def load_template_mesh(template_path: str) -> pv.PolyData:
    """Load template mesh for connectivity."""
    mesh = pv.read(template_path)
    print(f"[PCAExplorer] Loaded template: {mesh.n_points} points, {mesh.n_cells} cells")
    return mesh


# =============================================================================
# Deformation Grid
# =============================================================================

def create_deformation_grid(
    points: np.ndarray,
    resolution: int = 6,
    padding: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 3D wireframe grid around the shape bounding box.

    Returns:
        grid_points: (M, 3) grid vertices
        edges: (E, 2) edge indices
    """
    pmin, pmax = points.min(axis=0), points.max(axis=0)
    pad = (pmax - pmin) * padding
    bbox_min, bbox_max = pmin - pad, pmax + pad

    n = resolution + 1
    x = np.linspace(bbox_min[0], bbox_max[0], n)
    y = np.linspace(bbox_min[1], bbox_max[1], n)
    z = np.linspace(bbox_min[2], bbox_max[2], n)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    edges = []

    def idx(i, j, k):
        return i * n * n + j * n + k

    # Create edges only on boundary faces (wireframe box)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                on_face = (i in [0, n-1] or j in [0, n-1] or k in [0, n-1])
                if not on_face:
                    continue

                curr = idx(i, j, k)
                if i < n - 1 and (j in [0, n-1] or k in [0, n-1]):
                    edges.append([curr, idx(i + 1, j, k)])
                if j < n - 1 and (i in [0, n-1] or k in [0, n-1]):
                    edges.append([curr, idx(i, j + 1, k)])
                if k < n - 1 and (i in [0, n-1] or j in [0, n-1]):
                    edges.append([curr, idx(i, j, k + 1)])

    return grid_points, np.array(edges)


def deform_grid(
    grid_points: np.ndarray,
    source_shape: np.ndarray,
    target_shape: np.ndarray,
    kernel_scale: float,
) -> np.ndarray:
    """
    Interpolate shape deformation to grid points using Gaussian RBF.

    The displacement field on the shape is extended to ambient space
    via normalized Gaussian kernel interpolation.
    """
    displacement = target_shape - source_shape  # (N, 3)

    # Gaussian kernel weights
    diff = grid_points[:, np.newaxis, :] - source_shape[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    weights = np.exp(-dist_sq / (2 * kernel_scale**2))

    # Normalize
    weights_sum = weights.sum(axis=1, keepdims=True) + 1e-10
    weights_normalized = weights / weights_sum

    # Interpolate displacement
    grid_displacement = weights_normalized @ displacement

    return grid_points + grid_displacement


def grid_to_polydata(points: np.ndarray, edges: np.ndarray) -> pv.PolyData:
    """Convert grid points and edges to PyVista PolyData."""
    lines = np.zeros((len(edges), 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1:] = edges
    return pv.PolyData(points, lines=lines.ravel())


# =============================================================================
# Explorer
# =============================================================================

class PCAExplorer:
    """Interactive PCA shape explorer with deformation grid."""

    COLORS = {
        "mesh": "#E8D4B8",
        "background": "#1a1a2e",
        "text": "white",
        "grid": "#7799BB",
    }

    def __init__(
        self,
        model: PCAModel,
        template: pv.PolyData,
        n_components: int = 5,
        sigma_range: float = 3.0,
        window_size: Tuple[int, int] = (1400, 900),
        grid_resolution: int = 6,
        kernel_scale: Optional[float] = None,
    ):
        self.model = model
        self.template = template
        self.n_components = min(n_components, model.n_components)
        self.sigma_range = sigma_range
        self.window_size = window_size

        # State
        self.weights = np.zeros(self.n_components)
        self.heatmap_mode = False
        self.grid_mode = False

        # Create mesh
        self.mesh = self._create_mesh(model.mean)

        # Create deformation grid
        self.grid_resolution = grid_resolution
        self.grid_base_points, self.grid_edges = create_deformation_grid(
            model.mean, resolution=grid_resolution
        )
        self.grid_mesh = grid_to_polydata(self.grid_base_points, self.grid_edges)

        # Kernel scale for grid deformation (auto: ~10% of bbox diagonal)
        if kernel_scale is None:
            bbox_diag = np.linalg.norm(model.mean.max(axis=0) - model.mean.min(axis=0))
            self.kernel_scale = bbox_diag * 0.1
        else:
            self.kernel_scale = kernel_scale

        self.plotter: Optional[pv.Plotter] = None

    def _create_mesh(self, points: np.ndarray) -> pv.PolyData:
        """Create mesh with template connectivity."""
        mesh = self.template.copy()
        mesh.points = points
        return mesh

    def _update_display(self, _=None) -> None:
        """Update mesh and grid from current weights."""
        shape = generate_shape(self.model, self.weights)
        self.mesh.points = shape

        # Update heatmap scalars if active
        if self.heatmap_mode:
            deviation = np.linalg.norm(shape - self.model.mean, axis=1)
            self.mesh["Deviation (mm)"] = deviation

        # Update grid if visible
        if self.grid_mode:
            deformed_grid = deform_grid(
                self.grid_base_points,
                self.model.mean,
                shape,
                self.kernel_scale,
            )
            self.grid_mesh.points = deformed_grid

        if self.plotter is not None:
            self.plotter.render()

    def _toggle_grid(self) -> None:
        """Toggle deformation grid visibility."""
        self.grid_mode = not self.grid_mode
        self._refresh_display()
        print(f"[PCAExplorer] Grid: {'ON' if self.grid_mode else 'OFF'}")

    def _toggle_heatmap(self) -> None:
        """Toggle heatmap visualization mode."""
        self.heatmap_mode = not self.heatmap_mode
        self._refresh_display()
        print(f"[PCAExplorer] Heatmap: {'ON' if self.heatmap_mode else 'OFF'}")

    def _refresh_display(self) -> None:
        """Refresh all visual elements."""
        if self.plotter is None:
            return

        # Remove existing actors
        self.plotter.remove_actor("shape_mesh")
        self.plotter.remove_actor("deform_grid")

        # Add mesh
        if self.heatmap_mode:
            deviation = np.linalg.norm(self.mesh.points - self.model.mean, axis=1)
            self.mesh["Deviation (mm)"] = deviation
            self.plotter.add_mesh(
                self.mesh,
                scalars="Deviation (mm)",
                cmap="coolwarm",
                smooth_shading=True,
                name="shape_mesh",
                show_scalar_bar=True,
                scalar_bar_args={"title": "Deviation (mm)", "position_x": 0.05, "width": 0.25},
            )
        else:
            self.plotter.add_mesh(
                self.mesh,
                color=self.COLORS["mesh"],
                smooth_shading=True,
                name="shape_mesh",
            )

        # Add grid if enabled
        if self.grid_mode:
            # Update grid deformation
            shape = generate_shape(self.model, self.weights)
            deformed_grid = deform_grid(
                self.grid_base_points,
                self.model.mean,
                shape,
                self.kernel_scale,
            )
            self.grid_mesh.points = deformed_grid

            self.plotter.add_mesh(
                self.grid_mesh,
                color=self.COLORS["grid"],
                line_width=2,
                opacity=0.7,
                name="deform_grid",
            )

        self.plotter.render()

    def _reset_weights(self) -> None:
        """Reset all weights to zero."""
        self.weights[:] = 0
        self._update_display()

        if self.plotter is not None:
            for i, slider in enumerate(self.plotter.slider_widgets):
                if i < self.n_components:
                    slider.GetRepresentation().SetValue(0)
            self.plotter.render()

        print("[PCAExplorer] Reset to mean shape")

    def _make_slider_callback(self, idx: int):
        """Create callback for slider."""
        def callback(value):
            self.weights[idx] = value
            self._update_display()
        return callback

    def run(self) -> None:
        """Launch the interactive explorer."""
        self.plotter = pv.Plotter(
            window_size=self.window_size,
            title=f"PCA Explorer ({self.model.model_type.title()})",
        )
        self.plotter.set_background(self.COLORS["background"])

        # Add mesh
        self.plotter.add_mesh(
            self.mesh,
            color=self.COLORS["mesh"],
            smooth_shading=True,
            name="shape_mesh",
        )

        # Add axes
        self.plotter.add_axes()

        # Title
        model_label = "Tangent PCA" if self.model.model_type == "tangent" else "Linear PCA"
        self.plotter.add_text(
            f"{model_label} Shape Explorer",
            position="upper_edge",
            font_size=14,
            color=self.COLORS["text"],
        )

        # Controls help
        self.plotter.add_text(
            "G: Grid | H: Heatmap | R: Reset | Q: Quit",
            position="lower_left",
            font_size=10,
            color=self.COLORS["text"],
        )

        # Variance info
        var_lines = ["Variance Explained:"]
        cumsum = 0.0
        for i in range(min(5, self.n_components)):
            v = self.model.explained_variance_ratio[i] * 100
            cumsum += v
            var_lines.append(f"  PC{i+1}: {v:.1f}% (Σ={cumsum:.1f}%)")
        self.plotter.add_text(
            "\n".join(var_lines),
            position="upper_left",
            font_size=9,
            color=self.COLORS["text"],
        )

        # Sliders
        slider_top = 0.90
        slider_spacing = 0.10

        for i in range(self.n_components):
            var_pct = self.model.explained_variance_ratio[i] * 100
            self.plotter.add_slider_widget(
                callback=self._make_slider_callback(i),
                rng=(-self.sigma_range, self.sigma_range),
                value=0,
                title=f"PC{i+1} ({var_pct:.1f}%)",
                pointa=(0.72, slider_top - i * slider_spacing),
                pointb=(0.95, slider_top - i * slider_spacing),
                style="modern",
                title_color=self.COLORS["text"],
                fmt="%.2fσ",
            )

        # Key bindings
        self.plotter.add_key_event("r", lambda: self._reset_weights())
        self.plotter.add_key_event("R", lambda: self._reset_weights())
        self.plotter.add_key_event("h", lambda: self._toggle_heatmap())
        self.plotter.add_key_event("H", lambda: self._toggle_heatmap())
        self.plotter.add_key_event("g", lambda: self._toggle_grid())
        self.plotter.add_key_event("G", lambda: self._toggle_grid())

        # Camera
        self.plotter.enable_trackball_style()
        self.plotter.camera_position = "xy"
        self.plotter.camera.azimuth = -30
        self.plotter.camera.elevation = 20
        self.plotter.reset_camera()

        print("[PCAExplorer] Controls: Sliders=PC weights, G=Grid, H=Heatmap, R=Reset, Q=Quit")
        self.plotter.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive PCA Shape Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "-m", required=True, help="Path to PCA model directory")
    parser.add_argument("--template", "-t", required=True, help="Path to template OBJ file")
    parser.add_argument("--components", "-c", type=int, default=5, help="Number of components (default: 5)")
    parser.add_argument("--sigma", "-s", type=float, default=3.0, help="Slider range in σ (default: 3.0)")
    parser.add_argument("--grid-resolution", type=int, default=6, help="Grid resolution (default: 6)")
    parser.add_argument("--kernel-scale", type=float, default=None, help="Kernel scale for grid deformation")
    args = parser.parse_args()

    model = load_model(args.model)
    template = load_template_mesh(args.template)

    explorer = PCAExplorer(
        model=model,
        template=template,
        n_components=args.components,
        sigma_range=args.sigma,
        grid_resolution=args.grid_resolution,
        kernel_scale=args.kernel_scale,
    )
    explorer.run()


if __name__ == "__main__":
    main()
