#!/usr/bin/env python3
"""
Interactive PCA Shape Explorer

A GUI application for exploring PCA shape models interactively.
Features:
- Slider controls for each principal component
- Real-time shape updates
- Reset to mean shape
- Export current shape

Author: Femur Modeling Project
Date: 2026
"""

import os
import sys
import numpy as np
import pyvista as pv
from pyvista import themes

# Import from our visualizer module
from pca_visualizer import (
    load_pca_model, 
    load_template_mesh,
    generate_shape,
    shape_to_points,
    create_mesh_from_shape,
    PCAModel
)


class PCAExplorer:
    """
    Interactive GUI for exploring PCA shape models.
    
    Provides slider controls to manipulate principal component weights
    and visualize the resulting shape in real-time.
    
    Example:
        >>> explorer = PCAExplorer('bin/pca_model.bin', 'data/training/L_Femur_11.obj')
        >>> explorer.run()
    """
    
    def __init__(self, 
                 model_path: str, 
                 template_path: str,
                 n_sliders: int = 10,
                 sigma_range: float = 3.0):
        """
        Initialize the PCA explorer.
        
        Args:
            model_path: Path to PCA model .bin file
            template_path: Path to template OBJ file
            n_sliders: Number of PC sliders to show
            sigma_range: Range of sigma for sliders (±sigma_range)
        """
        print("[Explorer] Loading PCA model...")
        self.model = load_pca_model(model_path)
        
        print("[Explorer] Loading template mesh...")
        self.template = load_template_mesh(template_path)
        
        self.n_sliders = min(n_sliders, self.model.n_components)
        self.sigma_range = sigma_range
        
        # Current weights
        self.weights = np.zeros(self.model.n_components)
        
        # Create initial mesh
        self.mesh = create_mesh_from_shape(self.model.mean, self.template)
        
        # Plotter reference
        self.plotter = None
        
        # Compute variance info
        self.variance_ratios = self.model.variances / self.model.total_variance * 100
        self.cumulative_variance = np.cumsum(self.variance_ratios)
        
        print(f"[Explorer] Ready with {self.n_sliders} controllable components")
    
    def _update_mesh(self) -> None:
        """Update mesh based on current weights."""
        shape = generate_shape(self.model, self.weights)
        self.mesh.points = shape_to_points(shape)
    
    def _create_slider_callback(self, index: int):
        """Create a callback for a specific slider."""
        def callback(value):
            self.weights[index] = value
            self._update_mesh()
        return callback
    
    def _reset_callback(self) -> None:
        """Reset all weights to zero (mean shape)."""
        self.weights = np.zeros(self.model.n_components)
        self._update_mesh()
        
        # Reset slider positions (if plotter available)
        # Note: PyVista doesn't easily allow programmatic slider updates
        print("[Explorer] Reset to mean shape (restart to update sliders)")
    
    def _random_callback(self) -> None:
        """Set random weights (sampled from standard normal)."""
        self.weights[:self.n_sliders] = np.random.randn(self.n_sliders)
        self._update_mesh()
        print(f"[Explorer] Random weights set: {self.weights[:5]}...")
    
    def _export_callback(self) -> None:
        """Export current shape to OBJ file."""
        filename = f"exported_shape_{np.random.randint(10000):04d}.obj"
        self.mesh.save(filename)
        print(f"[Explorer] Shape exported: {filename}")
    
    def run(self, window_size=(1400, 900)):
        """
        Start the interactive explorer.
        
        Args:
            window_size: Window dimensions
        """
        # Set theme
        pv.set_plot_theme('dark')
        
        self.plotter = pv.Plotter(
            window_size=window_size,
            title='PCA Shape Explorer'
        )
        
        # Dark professional background
        self.plotter.set_background('#1a1a2e')
        
        # Add mesh
        self.plotter.add_mesh(
            self.mesh,
            color='#E8D4B8',  # Bone color
            smooth_shading=True,
            name='shape'
        )
        
        # --- Compact slider panel on the left ---
        slider_width = 0.25       # Narrower sliders
        slider_x = 0.02           # Left margin
        slider_spacing = 0.055    # Tighter vertical spacing
        slider_start_y = 0.92    # Start from top
        
        # Add variance header
        self.plotter.add_text(
            "Principal Components",
            position=(0.02, 0.95),
            font_size=11,
            color='white',
            font='arial'
        )
        
        for i in range(self.n_sliders):
            y_pos = slider_start_y - i * slider_spacing
            var_pct = self.variance_ratios[i]
            
            # Slider widget with PC label as title
            self.plotter.add_slider_widget(
                self._create_slider_callback(i),
                rng=[-self.sigma_range, self.sigma_range],
                value=0.0,
                title=f"PC{i+1}",
                pointa=(slider_x, y_pos),
                pointb=(slider_x + slider_width, y_pos),
                style='modern',
                title_height=0.012,
                title_color='white',
                fmt="%.1f"
            )
            
            # Add variance percentage as 2D text annotation to the right of slider
            self.plotter.add_text(
                f"{var_pct:4.1f}%",
                position=(slider_x + slider_width + 0.02, y_pos - 0.018),
                font_size=10,
                color='#CCCCCC',
                viewport=True
            )
        
        # --- Info panel in bottom-left ---
        cum_var = self.cumulative_variance[self.n_sliders - 1]
        info_text = (
            f"Model Info:\n"
            f"  Vertices: {self.model.n_dimensions // 3:,}\n"
            f"  Components: {self.model.n_components}\n"
            f"  Samples: {self.model.n_samples}\n"
            f"  Variance explained: {cum_var:.1f}%"
        )
        
        self.plotter.add_text(
            info_text,
            position=(0.02, 0.02),
            font_size=9,
            color='#888888',
            font='courier'
        )
        
        # --- Title in upper-right ---
        self.plotter.add_text(
            "PCA Shape Explorer",
            position='upper_right',
            font_size=14,
            color='white'
        )
        
        # --- Controls help in lower-right ---
        controls_text = (
            "Controls:\n"
            "  Drag sliders to modify shape\n"
            "  Mouse: rotate | Scroll: zoom\n"
            "  R: reset view | Q: quit"
        )
        
        self.plotter.add_text(
            controls_text,
            position=(0.75, 0.02),
            font_size=9,
            color='#666666',
            font='courier'
        )
        
        # Add axes
        self.plotter.add_axes()
        
        print("\n" + "=" * 50)
        print("    PCA Shape Explorer - Interactive Mode")
        print("=" * 50)
        print(f"  • {self.n_sliders} principal components loaded")
        print(f"  • Slider range: ±{self.sigma_range}σ")
        print(f"  • Press 'q' to quit")
        print("=" * 50 + "\n")
        
        self.plotter.show()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive PCA Shape Explorer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python pca_explorer.py --model ../bin/pca_femur_model.bin --template ../data/training/L_Femur_11_DECIM.obj.FINAL.obj

Controls:
  • Use sliders to adjust principal component weights
  • Each slider controls one mode of variation
  • Weights are in units of standard deviation (σ)
  • Press 'r' to reset camera view
  • Press 'q' to quit
        """
    )
    
    parser.add_argument(
        '--model', '-m', 
        default='../bin/pca_femur_model.bin',
        help='Path to PCA model .bin file'
    )
    parser.add_argument(
        '--template', '-t',
        default='../data/training/L_Femur_11_DECIM.obj.FINAL.obj',
        help='Path to template OBJ file'
    )
    parser.add_argument(
        '--sliders', '-s',
        type=int,
        default=10,
        help='Number of PC sliders (default: 10)'
    )
    parser.add_argument(
        '--range', '-r',
        type=float,
        default=3.0,
        help='Sigma range for sliders (default: ±3.0)'
    )
    
    args = parser.parse_args()
    
    try:
        explorer = PCAExplorer(
            model_path=args.model,
            template_path=args.template,
            n_sliders=args.sliders,
            sigma_range=args.range
        )
        explorer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure to provide valid paths to the model and template files.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
