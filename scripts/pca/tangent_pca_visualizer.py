#!/usr/bin/env python3
"""
Tangent PCA Shape Model Visualization Module

This module provides visualization tools for LDDMM-based Tangent PCA shape models.
It mirrors the interface of pca_visualizer.py for consistent user experience.

Features:
- Mean shape (atlas) visualization
- Principal geodesic modes of variation
- Mode animations along geodesics
- Variance analysis plots
- Comparison with Linear PCA

Author: Femur Modeling Project
Date: 2026
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add parent to path for imports (go up to project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lddmm.tangent_pca import TangentPCA
from lddmm.femur_lddmm.data_loader import FemurDataLoader


# =============================================================================
# Tangent PCA Model Wrapper (compatible interface with Linear PCA)
# =============================================================================

@dataclass
class TangentPCAModel:
    """
    Wrapper to provide a consistent interface with linear PCAModel.
    
    Attributes:
        atlas: Atlas shape (N, 3)
        mean_momentum: Mean momentum (N, 3)  
        components: Principal components (K, N, 3)
        variances: Eigenvalues/variance for each component (K,)
        n_dimensions: Number of dimensions (N*3)
        n_samples: Number of training samples
        n_components: Number of components (K)
        total_variance: Total variance in the data
        explained_variance_ratio: Variance ratio per component
    """
    atlas: np.ndarray
    mean_momentum: np.ndarray
    components: np.ndarray
    variances: np.ndarray
    n_dimensions: int
    n_samples: int
    n_components: int
    total_variance: float
    explained_variance_ratio: np.ndarray


def load_tangent_pca_model(model_dir: str) -> TangentPCAModel:
    """
    Load a Tangent PCA model from saved files.
    
    Args:
        model_dir: Directory containing saved Tangent PCA model
        
    Returns:
        TangentPCAModel object
    """
    pca = TangentPCA.load(model_dir)
    
    # Compute total variance from eigenvalues
    total_variance = float(np.sum(pca.eigenvalues))
    
    return TangentPCAModel(
        atlas=pca.atlas,
        mean_momentum=pca.mean_momentum,
        components=pca.components,
        variances=pca.eigenvalues,
        n_dimensions=pca.n_points * 3,
        n_samples=pca.n_samples,
        n_components=pca.n_components,
        total_variance=total_variance,
        explained_variance_ratio=pca.explained_variance_ratio
    )


# =============================================================================
# Shape Generation
# =============================================================================

def generate_tangent_shape(model: TangentPCAModel, weights: np.ndarray) -> np.ndarray:
    """
    Generate a shape from Tangent PCA weights.
    
    Shape = atlas + mean_momentum + sum_k(weight_k * sqrt(variance_k) * component_k)
    
    This is the linearized exponential map in the tangent space.
    
    Args:
        model: Tangent PCA model
        weights: Array of weights (typically in [-3, 3] for 3-sigma)
        
    Returns:
        Generated shape (N, 3)
    """
    n_weights = min(len(weights), model.n_components)
    
    # Start with atlas + mean momentum
    momentum = model.mean_momentum.copy()
    
    # Add weighted components
    for k in range(n_weights):
        std = np.sqrt(model.variances[k])
        momentum = momentum + weights[k] * std * model.components[k]
    
    # Linearized exponential map: shape = atlas + momentum
    return model.atlas + momentum


def generate_tangent_mode_variation(
    model: TangentPCAModel, 
    mode: int, 
    sigma: float
) -> np.ndarray:
    """
    Generate a shape along a single principal geodesic.
    
    Args:
        model: Tangent PCA model
        mode: Index of the mode (0-based)
        sigma: Number of standard deviations
        
    Returns:
        Generated shape (N, 3)
    """
    if mode >= model.n_components:
        raise ValueError(f"Mode {mode} out of range. Max: {model.n_components - 1}")
    
    std = np.sqrt(model.variances[mode])
    momentum = model.mean_momentum + sigma * std * model.components[mode]
    return model.atlas + momentum


# =============================================================================
# Mesh Utilities
# =============================================================================

def load_template_mesh(obj_path: str) -> pv.PolyData:
    """
    Load a template mesh to get face connectivity.
    
    Args:
        obj_path: Path to a template OBJ file
        
    Returns:
        PyVista PolyData mesh
    """
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Template mesh not found: {obj_path}")
    return pv.read(obj_path)


def create_mesh_from_points(points: np.ndarray, template: pv.PolyData) -> pv.PolyData:
    """
    Create a mesh from point coordinates using template connectivity.
    
    Args:
        points: (N, 3) point coordinates
        template: Template mesh with face connectivity
        
    Returns:
        New mesh with updated vertices
    """
    mesh = template.copy()
    mesh.points = points
    return mesh


# =============================================================================
# Tangent PCA Visualizer Class
# =============================================================================

class TangentPCAVisualizer:
    """
    Visualization class for Tangent PCA shape models.
    
    Mirrors the interface of PCAVisualizer for consistency.
    
    Example:
        >>> model = load_tangent_pca_model('model/tangent_pca')
        >>> template = load_template_mesh('data/training/L_Femur_11.obj')
        >>> viz = TangentPCAVisualizer(model, template)
        >>> viz.show_atlas()
        >>> viz.show_mode_variation(mode=0)
    """
    
    # Color scheme matching PCAVisualizer
    COLORS = {
        'atlas': '#E8D4B8',      # Warm bone color
        'positive': '#4A90D9',   # Blue for +sigma
        'negative': '#D94A4A',   # Red for -sigma
        'neutral': '#888888',    # Gray
        'background': '#1a1a2e', # Dark background
    }
    
    def __init__(self, model: TangentPCAModel, template: pv.PolyData):
        """
        Initialize the Tangent PCA visualizer.
        
        Args:
            model: Loaded Tangent PCA model
            template: Template mesh for face connectivity
        """
        self.model = model
        self.template = template
        self.n_vertices = model.atlas.shape[0]
        
        # Verify dimensions match
        if template.n_points != self.n_vertices:
            raise ValueError(
                f"Dimension mismatch: Model has {self.n_vertices} vertices, "
                f"template has {template.n_points}"
            )
        
        # Pre-compute variance info
        self.variance_ratios = model.explained_variance_ratio
        self.cumulative_variance = np.cumsum(self.variance_ratios)
        
        print(f"[Tangent PCA Visualizer] Initialized with {model.n_components} components")
        print(f"[Tangent PCA Visualizer] Template: {template.n_points} vertices, {template.n_cells} faces")
    
    def _create_mesh(self, points: np.ndarray) -> pv.PolyData:
        """Create a mesh from point coordinates."""
        return create_mesh_from_points(points, self.template)
    
    def _setup_plotter(self, 
                       title: str = "Tangent PCA Visualization",
                       window_size: Tuple[int, int] = (1400, 900),
                       background: str = None) -> pv.Plotter:
        """Create and configure a plotter."""
        plotter = pv.Plotter(window_size=window_size, title=title)
        plotter.set_background(background or self.COLORS['background'])
        return plotter
    
    # =========================================================================
    # Single Shape Visualization
    # =========================================================================
    
    def show_atlas(self,
                   window_size: Tuple[int, int] = (1200, 800),
                   color: str = None,
                   show_edges: bool = False,
                   screenshot: str = None) -> None:
        """
        Display the atlas (mean/Fréchet mean) shape.
        
        Args:
            window_size: Window dimensions
            color: Mesh color
            show_edges: Whether to show mesh edges
            screenshot: If provided, save screenshot to this path
        """
        # Atlas + mean momentum gives the "mean shape" in original space
        mean_shape = self.model.atlas + self.model.mean_momentum
        mesh = self._create_mesh(mean_shape)
        
        plotter = self._setup_plotter(
            title="Tangent PCA Atlas Shape",
            window_size=window_size
        )
        
        plotter.add_mesh(
            mesh,
            color=color or self.COLORS['atlas'],
            smooth_shading=True,
            show_edges=show_edges
        )
        
        plotter.add_text(
            "Atlas (Fréchet Mean)",
            position='upper_left',
            font_size=14,
            color='white'
        )
        
        plotter.add_text(
            f"Vertices: {self.n_vertices:,}\n"
            f"Components: {self.model.n_components}\n"
            f"Samples: {self.model.n_samples}",
            position='lower_left',
            font_size=10,
            color='white'
        )
        
        plotter.add_axes()
        
        if screenshot:
            plotter.show(screenshot=screenshot, auto_close=True)
            print(f"[Tangent PCA] Screenshot saved: {screenshot}")
        else:
            plotter.show()
    
    def show_mode_variation(self,
                            mode: int = 0,
                            sigma_range: Tuple[float, float] = (-2.0, 2.0),
                            n_steps: int = 5,
                            window_size: Tuple[int, int] = (1600, 600),
                            screenshot: str = None) -> None:
        """
        Display a principal geodesic with multiple sigma levels side by side.
        
        Args:
            mode: Principal component index (0-based)
            sigma_range: Range of standard deviations to show
            n_steps: Number of shapes to display
            window_size: Window dimensions
            screenshot: If provided, save screenshot to this path
        """
        if mode >= self.model.n_components:
            raise ValueError(f"Mode {mode} out of range")
        
        sigmas = np.linspace(sigma_range[0], sigma_range[1], n_steps)
        
        plotter = pv.Plotter(
            shape=(1, n_steps),
            window_size=window_size,
            title=f"PC{mode + 1} Principal Geodesic"
        )
        plotter.set_background(self.COLORS['background'])
        
        # Color gradient
        cmap = LinearSegmentedColormap.from_list(
            'mode_cmap',
            [self.COLORS['negative'], self.COLORS['neutral'], self.COLORS['positive']]
        )
        
        for i, sigma in enumerate(sigmas):
            plotter.subplot(0, i)
            
            shape = generate_tangent_mode_variation(self.model, mode, sigma)
            mesh = self._create_mesh(shape)
            
            # Color based on sigma
            norm_sigma = (sigma - sigma_range[0]) / (sigma_range[1] - sigma_range[0])
            color = cmap(norm_sigma)[:3]
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0]*255), int(color[1]*255), int(color[2]*255)
            )
            
            plotter.add_mesh(mesh, color=hex_color, smooth_shading=True)
            
            sign = '+' if sigma >= 0 else ''
            plotter.add_text(
                f"σ = {sign}{sigma:.1f}",
                position='upper_edge',
                font_size=12,
                color='white'
            )
        
        variance_pct = self.variance_ratios[mode] * 100
        cumulative_pct = self.cumulative_variance[mode] * 100
        
        plotter.add_text(
            f"PC{mode + 1}: {variance_pct:.1f}% variance (cumulative: {cumulative_pct:.1f}%)",
            position='upper_left',
            font_size=14,
            color='white',
            viewport=True
        )
        
        if screenshot:
            plotter.show(screenshot=screenshot, auto_close=True)
        else:
            plotter.show()
    
    def show_multiple_modes(self,
                            n_modes: int = 5,
                            sigma: float = 2.0,
                            window_size: Tuple[int, int] = (1800, 800),
                            screenshot: str = None) -> None:
        """
        Display multiple principal geodesics with ±sigma variations in a grid.
        
        Args:
            n_modes: Number of modes to display
            sigma: Standard deviation level
            window_size: Window dimensions
            screenshot: If provided, save screenshot to this path
        """
        n_modes = min(n_modes, self.model.n_components)
        
        plotter = pv.Plotter(
            shape=(3, n_modes),
            window_size=window_size,
            title=f"First {n_modes} Principal Geodesics (±{sigma}σ)"
        )
        plotter.set_background(self.COLORS['background'])
        
        colors = [self.COLORS['negative'], self.COLORS['atlas'], self.COLORS['positive']]
        labels = [f'-{sigma}σ', 'Mean', f'+{sigma}σ']
        sigmas = [-sigma, 0, sigma]
        
        for mode in range(n_modes):
            for row, (s, color, label) in enumerate(zip(sigmas, colors, labels)):
                plotter.subplot(row, mode)
                
                shape = generate_tangent_mode_variation(self.model, mode, s)
                mesh = self._create_mesh(shape)
                
                plotter.add_mesh(mesh, color=color, smooth_shading=True)
                
                if row == 0:
                    var_pct = self.variance_ratios[mode] * 100
                    plotter.add_text(
                        f"PC{mode + 1}\n({var_pct:.1f}%)",
                        position='upper_edge',
                        font_size=10,
                        color='white'
                    )
                
                if mode == 0:
                    plotter.add_text(
                        label,
                        position='left_edge',
                        font_size=10,
                        color='white'
                    )
        
        if screenshot:
            plotter.show(screenshot=screenshot, auto_close=True)
        else:
            plotter.show()
    
    # =========================================================================
    # Animation
    # =========================================================================
    
    def animate_mode(self,
                     mode: int = 0,
                     sigma_range: Tuple[float, float] = (-3.0, 3.0),
                     n_frames: int = 60,
                     fps: int = 30,
                     window_size: Tuple[int, int] = (1200, 800),
                     save_gif: str = None) -> None:
        """
        Animate a principal geodesic oscillating between ±sigma.
        
        Args:
            mode: Principal component index
            sigma_range: Range of standard deviations
            n_frames: Number of frames in animation
            fps: Frames per second
            window_size: Window dimensions
            save_gif: If provided, save animation as GIF
        """
        if mode >= self.model.n_components:
            raise ValueError(f"Mode {mode} out of range")
        
        # Generate sigma values (oscillating)
        t = np.linspace(0, 2 * np.pi, n_frames)
        sigmas = (sigma_range[0] + sigma_range[1]) / 2 + \
                 (sigma_range[1] - sigma_range[0]) / 2 * np.sin(t)
        
        # Create initial mesh
        initial_shape = generate_tangent_mode_variation(self.model, mode, sigmas[0])
        mesh = self._create_mesh(initial_shape)
        
        plotter = self._setup_plotter(
            title=f"PC{mode + 1} Geodesic Animation",
            window_size=window_size
        )
        
        plotter.add_mesh(mesh, color=self.COLORS['atlas'], smooth_shading=True)
        
        variance_pct = self.variance_ratios[mode] * 100
        
        plotter.add_text(
            f"σ = {sigmas[0]:+.2f}",
            position='upper_right',
            font_size=14,
            color='white',
            name='sigma_text'
        )
        
        plotter.add_text(
            f"PC{mode + 1}: {variance_pct:.1f}% variance",
            position='upper_left',
            font_size=14,
            color='white'
        )
        
        plotter.add_axes()
        
        if save_gif:
            plotter.open_gif(save_gif, fps=fps)
        
        def update_mesh(frame):
            sigma = sigmas[frame % n_frames]
            shape = generate_tangent_mode_variation(self.model, mode, sigma)
            mesh.points = shape
            
            plotter.remove_actor('sigma_text')
            plotter.add_text(
                f"σ = {sigma:+.2f}",
                position='upper_right',
                font_size=14,
                color='white',
                name='sigma_text'
            )
            
            if save_gif:
                plotter.write_frame()
        
        if save_gif:
            print(f"[Tangent PCA] Generating GIF: {save_gif}")
            for frame in range(n_frames):
                update_mesh(frame)
            plotter.close()
            print(f"[Tangent PCA] GIF saved: {save_gif}")
        else:
            plotter.add_callback(update_mesh, interval=1000 // fps)
            plotter.show()
    
    # =========================================================================
    # Statistical Plots
    # =========================================================================
    
    def plot_variance_explained(self,
                                 n_components: int = None,
                                 figsize: Tuple[int, int] = (12, 5),
                                 save_path: str = None) -> None:
        """
        Plot variance explained by each component and cumulative variance.
        """
        n_components = n_components or self.model.n_components
        n_components = min(n_components, self.model.n_components)
        
        indices = np.arange(1, n_components + 1)
        var_ratios = self.variance_ratios[:n_components] * 100
        cumulative = self.cumulative_variance[:n_components] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Tangent PCA Variance Analysis', fontsize=14, fontweight='bold')
        
        # Individual variance
        bars = ax1.bar(indices, var_ratios, color='#5D9CEC', edgecolor='#3A7BD5', alpha=0.8)
        ax1.set_xlabel('Principal Component', fontsize=11)
        ax1.set_ylabel('Variance Explained (%)', fontsize=11)
        ax1.set_title('Individual Variance per Component', fontsize=12)
        ax1.set_xticks(indices)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, var_ratios):
            if val > 2:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Cumulative variance
        ax2.plot(indices, cumulative, 'o-', color='#5D9CEC', linewidth=2, 
                markersize=6, markerfacecolor='white', markeredgewidth=2)
        ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90%')
        ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95%')
        ax2.axhline(y=99, color='green', linestyle='--', alpha=0.7, label='99%')
        
        ax2.set_xlabel('Number of Components', fontsize=11)
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=11)
        ax2.set_title('Cumulative Variance Explained', fontsize=12)
        ax2.set_xticks(indices)
        ax2.set_ylim(0, 105)
        ax2.grid(alpha=0.3)
        ax2.legend(loc='lower right')
        
        for threshold in [90, 95, 99]:
            k = np.searchsorted(cumulative, threshold) + 1
            if k <= n_components:
                ax2.annotate(f'K={k}', xy=(k, cumulative[k-1]), 
                           xytext=(k + 0.5, cumulative[k-1] - 5),
                           fontsize=9, color='darkred')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Tangent PCA] Figure saved: {save_path}")
        else:
            plt.show()
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def export_atlas(self, output_path: str) -> None:
        """Export the atlas shape as an OBJ file."""
        mean_shape = self.model.atlas + self.model.mean_momentum
        mesh = self._create_mesh(mean_shape)
        mesh.save(output_path)
        print(f"[Tangent PCA] Atlas exported: {output_path}")
    
    def export_mode_variations(self,
                                output_dir: str,
                                n_modes: int = 5,
                                sigmas: List[float] = [-2, -1, 0, 1, 2]) -> None:
        """
        Export mode variations as OBJ files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.export_atlas(os.path.join(output_dir, 'atlas_shape.obj'))
        
        n_modes = min(n_modes, self.model.n_components)
        
        for mode in range(n_modes):
            for sigma in sigmas:
                shape = generate_tangent_mode_variation(self.model, mode, sigma)
                mesh = self._create_mesh(shape)
                
                sign = 'plus' if sigma >= 0 else 'minus'
                filename = f'tangent_PC{mode + 1}_{sign}{abs(sigma):.0f}sigma.obj'
                filepath = os.path.join(output_dir, filename)
                
                mesh.save(filepath)
        
        print(f"[Tangent PCA] Exported {n_modes} modes to: {output_dir}")
    
    def generate_report(self, output_dir: str) -> None:
        """Generate a complete visualization report."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("[Tangent PCA] Generating visualization report...")
        
        self.plot_variance_explained(
            save_path=os.path.join(output_dir, 'tangent_variance_explained.png')
        )
        
        self.export_mode_variations(
            output_dir=os.path.join(output_dir, 'shapes'),
            n_modes=5,
            sigmas=[-2, -1, 0, 1, 2]
        )
        
        try:
            for mode in range(min(3, self.model.n_components)):
                self.show_mode_variation(
                    mode=mode,
                    screenshot=os.path.join(output_dir, f'tangent_PC{mode + 1}_variation.png')
                )
        except Exception as e:
            print(f"[Tangent PCA] Could not generate screenshots: {e}")
        
        print(f"[Tangent PCA] Report generated in: {output_dir}")


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_tangent_visualize(model_dir: str, template_path: str) -> TangentPCAVisualizer:
    """
    Quick setup for Tangent PCA visualization.
    
    Args:
        model_dir: Directory with saved Tangent PCA model
        template_path: Path to template OBJ file
        
    Returns:
        Configured TangentPCAVisualizer instance
    """
    model = load_tangent_pca_model(model_dir)
    template = load_template_mesh(template_path)
    return TangentPCAVisualizer(model, template)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tangent PCA Shape Model Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show atlas shape
  python tangent_pca_visualizer.py --model model/tangent_pca --template data/training/L_Femur_11.obj --atlas
  
  # Show mode variation
  python tangent_pca_visualizer.py --model model/tangent_pca --template data/training/L_Femur_11.obj --mode 0
  
  # Animate a mode
  python tangent_pca_visualizer.py --model model/tangent_pca --template data/training/L_Femur_11.obj --animate 0
  
  # Generate full report
  python tangent_pca_visualizer.py --model model/tangent_pca --template data/training/L_Femur_11.obj --report output/
        """
    )
    
    parser.add_argument('--model', '-m', required=True, help='Path to Tangent PCA model directory')
    parser.add_argument('--template', '-t', required=True, help='Path to template OBJ file')
    parser.add_argument('--atlas', action='store_true', help='Show atlas shape')
    parser.add_argument('--mode', type=int, help='Show mode variation (0-indexed)')
    parser.add_argument('--modes', type=int, help='Show first N modes in grid')
    parser.add_argument('--animate', type=int, help='Animate mode (0-indexed)')
    parser.add_argument('--variance', action='store_true', help='Show variance plots')
    parser.add_argument('--report', type=str, help='Generate full report to directory')
    parser.add_argument('--export', type=str, help='Export shapes to directory')
    
    args = parser.parse_args()
    
    viz = quick_tangent_visualize(args.model, args.template)
    
    if args.atlas:
        viz.show_atlas()
    
    if args.mode is not None:
        viz.show_mode_variation(mode=args.mode)
    
    if args.modes:
        viz.show_multiple_modes(n_modes=args.modes)
    
    if args.animate is not None:
        viz.animate_mode(mode=args.animate)
    
    if args.variance:
        viz.plot_variance_explained()
    
    if args.report:
        viz.generate_report(args.report)
    
    if args.export:
        viz.export_mode_variations(args.export)
    
    if not any([args.atlas, args.mode is not None, args.modes, args.animate is not None,
                args.variance, args.report, args.export]):
        parser.print_help()
