#!/usr/bin/env python3
"""
PCA Shape Model Visualization Module

This module provides comprehensive visualization tools for PCA-based
Statistical Shape Models of femur bones. It enables visualization of:
- Mean shape
- Principal component modes of variation
- Shape reconstruction quality
- Variance analysis plots

Author: Femur Modeling Project
Date: 2026
"""

import os
import sys
import struct
import numpy as np
import pyvista as pv
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Data Classes
# =============================================================================

# Standardization factors used in C++ Femur::getCoordsVect
STANDARDIZATION_FACTORS = {
    'x': 152.2,
    'y': 20.4,
    'z': 16.2
}

@dataclass
class PCAModel:
    """
    Data class holding a loaded PCA model.
    
    Attributes:
        mean: Mean shape vector (D,)
        components: Principal components matrix (D, K)
        variances: Variance for each component (K,)
        n_dimensions: Number of dimensions (D)
        n_samples: Number of training samples
        n_components: Number of components (K)
        total_variance: Total variance in the data
    """
    mean: np.ndarray
    components: np.ndarray
    variances: np.ndarray
    n_dimensions: int
    n_samples: int
    n_components: int
    total_variance: float


# =============================================================================
# PCA Model I/O
# =============================================================================

def load_pca_model(filepath: str) -> PCAModel:
    """
    Load a PCA model from binary file (saved by C++ PCA class).
    
    Args:
        filepath: Path to the .bin file
        
    Returns:
        PCAModel object with all model parameters
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PCA model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        # Read and verify magic header
        magic = f.read(4).decode('ascii')
        if magic != 'PCA1':
            raise ValueError(f"Invalid PCA file format. Expected 'PCA1', got '{magic}'")
        
        # Read metadata (size_t = 8 bytes on 64-bit systems)
        n_dimensions = struct.unpack('Q', f.read(8))[0]
        n_samples = struct.unpack('Q', f.read(8))[0]
        n_components = struct.unpack('Q', f.read(8))[0]
        total_variance = struct.unpack('d', f.read(8))[0]
        
        # Read mean vector
        mean = np.array(struct.unpack(f'{n_dimensions}d', f.read(n_dimensions * 8)))
        
        # Read variances
        variances = np.array(struct.unpack(f'{n_components}d', f.read(n_components * 8)))
        
        # Read components (column-major order)
        components_flat = np.array(struct.unpack(
            f'{n_dimensions * n_components}d', 
            f.read(n_dimensions * n_components * 8)
        ))
        components = components_flat.reshape((n_components, n_dimensions)).T
    
    print(f"[PCA] Loaded model: D={n_dimensions}, N={n_samples}, K={n_components}")
    
    return PCAModel(
        mean=mean,
        components=components,
        variances=variances,
        n_dimensions=n_dimensions,
        n_samples=n_samples,
        n_components=n_components,
        total_variance=total_variance
    )


# =============================================================================
# Shape Generation
# =============================================================================

def generate_shape(model: PCAModel, weights: np.ndarray) -> np.ndarray:
    """
    Generate a shape from PCA weights.
    
    Shape = mean + sum_k(weight_k * sqrt(variance_k) * component_k)
    
    Args:
        model: PCA model
        weights: Array of weights (typically in [-3, 3] for 3-sigma)
        
    Returns:
        Generated shape vector (D,)
    """
    n_weights = min(len(weights), model.n_components)
    shape = model.mean.copy()
    
    for k in range(n_weights):
        shape += weights[k] * np.sqrt(model.variances[k]) * model.components[:, k]
    
    return shape


def generate_mode_variation(model: PCAModel, mode: int, sigma: float) -> np.ndarray:
    """
    Generate a shape along a single principal component mode.
    
    Args:
        model: PCA model
        mode: Index of the mode (0-based)
        sigma: Number of standard deviations
        
    Returns:
        Generated shape vector (D,)
    """
    if mode >= model.n_components:
        raise ValueError(f"Mode {mode} out of range. Max: {model.n_components - 1}")
    
    std = np.sqrt(model.variances[mode])
    return model.mean + sigma * std * model.components[:, mode]


# =============================================================================
# Mesh Utilities
# =============================================================================

def shape_to_points(shape: np.ndarray, destandardize: bool = True) -> np.ndarray:
    """
    Convert a flattened shape vector to (N, 3) point array.
    
    The C++ code stores coordinates as [all_X, all_Y, all_Z] (stacked by axis),
    and applies standardization. This function reverses both operations.
    
    Args:
        shape: Flattened shape vector [x1,x2,...,xN, y1,y2,...,yN, z1,z2,...,zN]
        destandardize: Whether to multiply by standardization factors
        
    Returns:
        Points array of shape (N, 3) with interleaved [x,y,z] per vertex
    """
    n_vertices = len(shape) // 3
    
    # Extract X, Y, Z blocks (C++ stores as [all_X, all_Y, all_Z])
    x_coords = shape[0:n_vertices]
    y_coords = shape[n_vertices:2*n_vertices]
    z_coords = shape[2*n_vertices:3*n_vertices]
    
    # De-standardize if needed (multiply back by the factors used in C++)
    if destandardize:
        x_coords = x_coords * STANDARDIZATION_FACTORS['x']
        y_coords = y_coords * STANDARDIZATION_FACTORS['y']
        z_coords = z_coords * STANDARDIZATION_FACTORS['z']
    
    # Stack into (N, 3) array
    points = np.column_stack([x_coords, y_coords, z_coords])
    
    return points


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


def create_mesh_from_shape(shape: np.ndarray, template: pv.PolyData) -> pv.PolyData:
    """
    Create a mesh from a shape vector using template connectivity.
    
    Args:
        shape: Shape vector (D,)
        template: Template mesh with face connectivity
        
    Returns:
        New mesh with updated vertices
    """
    points = shape_to_points(shape)
    mesh = template.copy()
    mesh.points = points
    return mesh


# =============================================================================
# PCA Visualizer Class
# =============================================================================

class PCAVisualizer:
    """
    Comprehensive visualization class for PCA shape models.
    
    Provides methods to visualize:
    - Mean shape
    - Individual modes of variation
    - Mode animations
    - Multiple shapes side-by-side
    - Variance analysis plots
    
    Example:
        >>> model = load_pca_model('models/pca_model.bin')
        >>> template = load_template_mesh('data/training/L_Femur_11.obj')
        >>> viz = PCAVisualizer(model, template)
        >>> viz.show_mean_shape()
        >>> viz.show_mode_variation(mode=0)
        >>> viz.animate_mode(mode=0)
    """
    
    # Color scheme for visualizations
    COLORS = {
        'mean': '#E8D4B8',      # Warm bone color
        'positive': '#4A90D9',   # Blue for +sigma
        'negative': '#D94A4A',   # Red for -sigma
        'neutral': '#888888',    # Gray
        'background': '#1a1a2e', # Dark background
    }
    
    def __init__(self, model: PCAModel, template: pv.PolyData):
        """
        Initialize the PCA visualizer.
        
        Args:
            model: Loaded PCA model
            template: Template mesh for face connectivity
        """
        self.model = model
        self.template = template
        self.n_vertices = model.n_dimensions // 3
        
        # Verify dimensions match
        if template.n_points != self.n_vertices:
            raise ValueError(
                f"Dimension mismatch: PCA has {self.n_vertices} vertices, "
                f"template has {template.n_points}"
            )
        
        # Pre-compute explained variance ratios
        self.variance_ratios = model.variances / model.total_variance
        self.cumulative_variance = np.cumsum(self.variance_ratios)
        
        print(f"[PCA Visualizer] Initialized with {model.n_components} components")
        print(f"[PCA Visualizer] Template: {template.n_points} vertices, {template.n_cells} faces")
    
    def _create_mesh(self, shape: np.ndarray) -> pv.PolyData:
        """Create a mesh from a shape vector."""
        return create_mesh_from_shape(shape, self.template)
    
    def _setup_plotter(self, 
                       title: str = "PCA Visualization",
                       window_size: Tuple[int, int] = (1400, 900),
                       background: str = None) -> pv.Plotter:
        """Create and configure a plotter."""
        plotter = pv.Plotter(
            window_size=window_size,
            title=title
        )
        plotter.set_background(background or self.COLORS['background'])
        return plotter
    
    # =========================================================================
    # Single Shape Visualization
    # =========================================================================
    
    def show_mean_shape(self,
                        window_size: Tuple[int, int] = (1200, 800),
                        color: str = None,
                        show_edges: bool = False,
                        screenshot: str = None) -> None:
        """
        Display the mean shape.
        
        Args:
            window_size: Window dimensions
            color: Mesh color (default: bone color)
            show_edges: Whether to show mesh edges
            screenshot: If provided, save screenshot to this path
        """
        mesh = self._create_mesh(self.model.mean)
        
        plotter = self._setup_plotter(
            title="PCA Mean Shape",
            window_size=window_size
        )
        
        plotter.add_mesh(
            mesh,
            color=color or self.COLORS['mean'],
            smooth_shading=True,
            show_edges=show_edges
        )
        
        # Add text annotation
        plotter.add_text(
            "Mean Shape (μ)",
            position='upper_left',
            font_size=14,
            color='white'
        )
        
        plotter.add_text(
            f"Vertices: {self.n_vertices:,}\nComponents: {self.model.n_components}",
            position='lower_left',
            font_size=10,
            color='white'
        )
        
        plotter.add_axes()
        
        if screenshot:
            plotter.show(screenshot=screenshot, auto_close=True)
            print(f"[PCA] Screenshot saved: {screenshot}")
        else:
            plotter.show()
    
    def show_mode_variation(self,
                            mode: int = 0,
                            sigma_range: Tuple[float, float] = (-2.0, 2.0),
                            n_steps: int = 5,
                            window_size: Tuple[int, int] = (1600, 600),
                            screenshot: str = None) -> None:
        """
        Display a mode of variation with multiple sigma levels side by side.
        
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
        
        # Create subplots
        plotter = pv.Plotter(
            shape=(1, n_steps),
            window_size=window_size,
            title=f"PC{mode + 1} Mode of Variation"
        )
        plotter.set_background(self.COLORS['background'])
        
        # Color gradient from red (-) to gray (0) to blue (+)
        cmap = LinearSegmentedColormap.from_list(
            'mode_cmap',
            [self.COLORS['negative'], self.COLORS['neutral'], self.COLORS['positive']]
        )
        
        for i, sigma in enumerate(sigmas):
            plotter.subplot(0, i)
            
            shape = generate_mode_variation(self.model, mode, sigma)
            mesh = self._create_mesh(shape)
            
            # Color based on sigma
            norm_sigma = (sigma - sigma_range[0]) / (sigma_range[1] - sigma_range[0])
            color = cmap(norm_sigma)[:3]
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0]*255), int(color[1]*255), int(color[2]*255)
            )
            
            plotter.add_mesh(
                mesh,
                color=hex_color,
                smooth_shading=True
            )
            
            # Label
            sign = '+' if sigma >= 0 else ''
            plotter.add_text(
                f"σ = {sign}{sigma:.1f}",
                position='upper_edge',
                font_size=12,
                color='white'
            )
        
        # Add overall title
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
            print(f"[PCA] Screenshot saved: {screenshot}")
        else:
            plotter.show()
    
    def show_multiple_modes(self,
                            n_modes: int = 5,
                            sigma: float = 2.0,
                            window_size: Tuple[int, int] = (1800, 800),
                            screenshot: str = None) -> None:
        """
        Display multiple modes with ±sigma variations in a grid.
        
        Args:
            n_modes: Number of modes to display
            sigma: Standard deviation level
            window_size: Window dimensions
            screenshot: If provided, save screenshot to this path
        """
        n_modes = min(n_modes, self.model.n_components)
        
        # Grid: 3 rows (−σ, mean, +σ) × n_modes columns
        plotter = pv.Plotter(
            shape=(3, n_modes),
            window_size=window_size,
            title=f"First {n_modes} Principal Components (±{sigma}σ)"
        )
        plotter.set_background(self.COLORS['background'])
        
        colors = [self.COLORS['negative'], self.COLORS['mean'], self.COLORS['positive']]
        labels = [f'-{sigma}σ', 'Mean', f'+{sigma}σ']
        sigmas = [-sigma, 0, sigma]
        
        for mode in range(n_modes):
            for row, (s, color, label) in enumerate(zip(sigmas, colors, labels)):
                plotter.subplot(row, mode)
                
                shape = generate_mode_variation(self.model, mode, s)
                mesh = self._create_mesh(shape)
                
                plotter.add_mesh(mesh, color=color, smooth_shading=True)
                
                # Column header (only on first row)
                if row == 0:
                    var_pct = self.variance_ratios[mode] * 100
                    plotter.add_text(
                        f"PC{mode + 1}\n({var_pct:.1f}%)",
                        position='upper_edge',
                        font_size=10,
                        color='white'
                    )
                
                # Row label (only on first column)
                if mode == 0:
                    plotter.add_text(
                        label,
                        position='left_edge',
                        font_size=10,
                        color='white'
                    )
        
        if screenshot:
            plotter.show(screenshot=screenshot, auto_close=True)
            print(f"[PCA] Screenshot saved: {screenshot}")
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
        Animate a mode of variation oscillating between ±sigma.
        
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
        initial_shape = generate_mode_variation(self.model, mode, sigmas[0])
        mesh = self._create_mesh(initial_shape)
        
        plotter = self._setup_plotter(
            title=f"PC{mode + 1} Animation",
            window_size=window_size
        )
        
        actor = plotter.add_mesh(
            mesh,
            color=self.COLORS['mean'],
            smooth_shading=True
        )
        
        variance_pct = self.variance_ratios[mode] * 100
        
        # Text actors that will be updated
        sigma_text = plotter.add_text(
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
            shape = generate_mode_variation(self.model, mode, sigma)
            points = shape_to_points(shape)
            mesh.points = points
            
            # Update sigma text
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
            print(f"[PCA] Generating GIF: {save_gif}")
            for frame in range(n_frames):
                update_mesh(frame)
            plotter.close()
            print(f"[PCA] GIF saved: {save_gif}")
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
        
        Args:
            n_components: Number of components to show (default: all)
            figsize: Figure size
            save_path: If provided, save figure to this path
        """
        n_components = n_components or self.model.n_components
        n_components = min(n_components, self.model.n_components)
        
        indices = np.arange(1, n_components + 1)
        var_ratios = self.variance_ratios[:n_components] * 100
        cumulative = self.cumulative_variance[:n_components] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('PCA Variance Analysis', fontsize=14, fontweight='bold')
        
        # Individual variance
        bars = ax1.bar(indices, var_ratios, color='steelblue', edgecolor='navy', alpha=0.8)
        ax1.set_xlabel('Principal Component', fontsize=11)
        ax1.set_ylabel('Variance Explained (%)', fontsize=11)
        ax1.set_title('Individual Variance per Component', fontsize=12)
        ax1.set_xticks(indices)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, var_ratios):
            if val > 2:  # Only label if > 2%
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Cumulative variance
        ax2.plot(indices, cumulative, 'o-', color='steelblue', linewidth=2, 
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
        
        # Annotate key thresholds
        for threshold in [90, 95, 99]:
            k = np.searchsorted(cumulative, threshold) + 1
            if k <= n_components:
                ax2.annotate(f'K={k}', xy=(k, cumulative[k-1]), 
                           xytext=(k + 0.5, cumulative[k-1] - 5),
                           fontsize=9, color='darkred')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[PCA] Figure saved: {save_path}")
        else:
            plt.show()
    
    def plot_mode_spectrum(self,
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: str = None) -> None:
        """
        Plot the eigenvalue spectrum (scree plot).
        
        Args:
            figsize: Figure size
            save_path: If provided, save figure to this path
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        indices = np.arange(1, self.model.n_components + 1)
        eigenvalues = self.model.variances
        
        ax.semilogy(indices, eigenvalues, 'o-', color='steelblue', 
                   linewidth=2, markersize=8)
        
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Eigenvalue (log scale)', fontsize=12)
        ax.set_title('Eigenvalue Spectrum (Scree Plot)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(indices)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[PCA] Figure saved: {save_path}")
        else:
            plt.show()
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def export_mean_shape(self, output_path: str) -> None:
        """Export the mean shape as an OBJ file."""
        mesh = self._create_mesh(self.model.mean)
        mesh.save(output_path)
        print(f"[PCA] Mean shape exported: {output_path}")
    
    def export_mode_variations(self,
                                output_dir: str,
                                n_modes: int = 5,
                                sigmas: List[float] = [-2, -1, 0, 1, 2]) -> None:
        """
        Export mode variations as OBJ files.
        
        Args:
            output_dir: Output directory
            n_modes: Number of modes to export
            sigmas: Sigma levels to export for each mode
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export mean
        self.export_mean_shape(os.path.join(output_dir, 'mean_shape.obj'))
        
        # Export modes
        n_modes = min(n_modes, self.model.n_components)
        
        for mode in range(n_modes):
            for sigma in sigmas:
                shape = generate_mode_variation(self.model, mode, sigma)
                mesh = self._create_mesh(shape)
                
                sign = 'plus' if sigma >= 0 else 'minus'
                filename = f'PC{mode + 1}_{sign}{abs(sigma):.0f}sigma.obj'
                filepath = os.path.join(output_dir, filename)
                
                mesh.save(filepath)
        
        print(f"[PCA] Exported {n_modes} modes to: {output_dir}")
    
    def generate_report(self, output_dir: str) -> None:
        """
        Generate a complete visualization report.
        
        Creates:
        - Variance analysis plots
        - Mode variation screenshots
        - Exported OBJ files
        
        Args:
            output_dir: Output directory for all files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("[PCA] Generating visualization report...")
        
        # 1. Variance plots
        self.plot_variance_explained(
            save_path=os.path.join(output_dir, 'variance_explained.png')
        )
        self.plot_mode_spectrum(
            save_path=os.path.join(output_dir, 'eigenvalue_spectrum.png')
        )
        
        # 2. Export OBJ files
        self.export_mode_variations(
            output_dir=os.path.join(output_dir, 'shapes'),
            n_modes=5,
            sigmas=[-2, -1, 0, 1, 2]
        )
        
        # 3. Generate mode visualizations (if display available)
        try:
            for mode in range(min(3, self.model.n_components)):
                self.show_mode_variation(
                    mode=mode,
                    screenshot=os.path.join(output_dir, f'PC{mode + 1}_variation.png')
                )
        except Exception as e:
            print(f"[PCA] Could not generate screenshots: {e}")
        
        print(f"[PCA] Report generated in: {output_dir}")


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_visualize(model_path: str, template_path: str) -> PCAVisualizer:
    """
    Quick setup for PCA visualization.
    
    Args:
        model_path: Path to PCA model .bin file
        template_path: Path to template OBJ file
        
    Returns:
        Configured PCAVisualizer instance
    """
    model = load_pca_model(model_path)
    template = load_template_mesh(template_path)
    return PCAVisualizer(model, template)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PCA Shape Model Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show mean shape
  python pca_visualizer.py --model bin/pca_model.bin --template data/training/L_Femur_11.obj --mean
  
  # Show mode variation
  python pca_visualizer.py --model bin/pca_model.bin --template data/training/L_Femur_11.obj --mode 0
  
  # Animate a mode
  python pca_visualizer.py --model bin/pca_model.bin --template data/training/L_Femur_11.obj --animate 0
  
  # Generate full report
  python pca_visualizer.py --model bin/pca_model.bin --template data/training/L_Femur_11.obj --report output/
        """
    )
    
    parser.add_argument('--model', '-m', required=True, help='Path to PCA model .bin file')
    parser.add_argument('--template', '-t', required=True, help='Path to template OBJ file')
    parser.add_argument('--mean', action='store_true', help='Show mean shape')
    parser.add_argument('--mode', type=int, help='Show mode variation (0-indexed)')
    parser.add_argument('--modes', type=int, help='Show first N modes in grid')
    parser.add_argument('--animate', type=int, help='Animate mode (0-indexed)')
    parser.add_argument('--variance', action='store_true', help='Show variance plots')
    parser.add_argument('--report', type=str, help='Generate full report to directory')
    parser.add_argument('--export', type=str, help='Export shapes to directory')
    
    args = parser.parse_args()
    
    # Load model and create visualizer
    viz = quick_visualize(args.model, args.template)
    
    # Execute requested visualizations
    if args.mean:
        viz.show_mean_shape()
    
    if args.mode is not None:
        viz.show_mode_variation(mode=args.mode)
    
    if args.modes:
        viz.show_multiple_modes(n_modes=args.modes)
    
    if args.animate is not None:
        viz.animate_mode(mode=args.animate)
    
    if args.variance:
        viz.plot_variance_explained()
        viz.plot_mode_spectrum()
    
    if args.report:
        viz.generate_report(args.report)
    
    if args.export:
        viz.export_mode_variations(args.export)
    
    # If no specific action, show help
    if not any([args.mean, args.mode is not None, args.modes, args.animate is not None,
                args.variance, args.report, args.export]):
        parser.print_help()
