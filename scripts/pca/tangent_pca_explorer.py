#!/usr/bin/env python3
"""
Tangent PCA Interactive Shape Explorer

Interactive 3D visualization tool for exploring LDDMM-based Tangent PCA shape models.
Uses slider widgets to navigate the principal geodesics in real-time.

This mirrors the interface of pca_explorer.py for consistent user experience.

Features:
- Real-time shape deformation along principal geodesics
- Slider controls for each principal component
- Variance explained display
- Reset functionality

Author: Femur Modeling Project
Date: 2026
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
from pyvista import Plotter

# Add parent directory to path for imports (project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pca.tangent_pca_visualizer import (
    TangentPCAModel,
    load_tangent_pca_model,
    load_template_mesh,
    generate_tangent_shape,
)


class TangentPCAExplorer:
    """
    Interactive explorer for Tangent PCA shape models.
    
    Uses slider widgets to control each principal component weight in real-time,
    allowing users to explore the shape space defined by the tangent PCA model.
    
    Example:
        >>> explorer = TangentPCAExplorer(
        ...     model_dir='model/tangent_pca',
        ...     template_path='data/training/L_Femur_11.obj'
        ... )
        >>> explorer.run()
    """
    
    # Color scheme
    COLORS = {
        'mesh': '#E8D4B8',        # Warm bone color
        'background': '#1a1a2e',  # Dark background
        'text': 'white',
        'positive': '#4A90D9',    # Blue
        'negative': '#D94A4A',    # Red
    }
    
    def __init__(self,
                 model_dir: str,
                 template_path: str,
                 n_components: int = 5,
                 sigma_range: float = 3.0,
                 window_size: Tuple[int, int] = (1400, 900)):
        """
        Initialize the Tangent PCA explorer.
        
        Args:
            model_dir: Path to directory containing saved Tangent PCA model
            template_path: Path to template OBJ file for mesh connectivity
            n_components: Number of principal components to control
            sigma_range: Range of standard deviations for sliders (±sigma_range)
            window_size: Window dimensions (width, height)
        """
        print(f"[Tangent PCA Explorer] Loading model from: {model_dir}")
        self.model = load_tangent_pca_model(model_dir)
        
        print(f"[Tangent PCA Explorer] Loading template from: {template_path}")
        self.template = load_template_mesh(template_path)
        
        # Setup parameters
        self.n_components = min(n_components, self.model.n_components)
        self.sigma_range = sigma_range
        self.window_size = window_size
        
        # Current weights (all zero = mean shape)
        self.weights = np.zeros(self.n_components)
        
        # Create initial mesh
        self.mean_shape = self.model.atlas + self.model.mean_momentum
        self.mesh = self._create_mesh(self.mean_shape)
        
        # Plotter will be created in run()
        self.plotter: Optional[Plotter] = None
        
        print(f"[Tangent PCA Explorer] Initialized with {self.n_components} controllable components")
        print(f"[Tangent PCA Explorer] Model: {self.model.n_samples} samples, "
              f"{self.model.atlas.shape[0]} vertices")
    
    def _create_mesh(self, points: np.ndarray) -> pv.PolyData:
        """Create a mesh from point coordinates using template connectivity."""
        mesh = self.template.copy()
        mesh.points = points
        return mesh
    
    def _update_mesh(self, value: float = None) -> None:
        """Update mesh based on current weights."""
        shape = generate_tangent_shape(self.model, self.weights)
        self.mesh.points = shape
        
        # Update the render
        if self.plotter is not None:
            self.plotter.render()
    
    def _create_slider_callback(self, component_idx: int):
        """Create a callback function for a specific component slider."""
        def callback(value: float) -> None:
            self.weights[component_idx] = value
            self._update_mesh()
        return callback
    
    def _reset_weights(self) -> None:
        """Reset all weights to zero."""
        self.weights[:] = 0
        self._update_mesh()
        
        # Update slider positions
        if self.plotter is not None:
            for i in range(self.n_components):
                slider = self.plotter.slider_widgets[i]
                slider.GetRepresentation().SetValue(0)
            self.plotter.render()
    
    def _get_controls_text(self) -> str:
        """Generate controls help text."""
        lines = [
            "╔══════════════════════════════════╗",
            "║     TANGENT PCA EXPLORER         ║",
            "╠══════════════════════════════════╣",
            "║  Sliders: Adjust PC weights      ║",
            "║  R: Reset all to mean shape      ║",
            "║  Q: Quit                         ║",
            "╠══════════════════════════════════╣",
            "║  Variance Explained:             ║",
        ]
        
        cumulative = 0
        for i in range(min(self.n_components, 5)):
            var = self.model.explained_variance_ratio[i] * 100
            cumulative += var
            lines.append(f"║  PC{i+1}: {var:5.1f}% (cum: {cumulative:5.1f}%)  ║")
        
        lines.append("╚══════════════════════════════════╝")
        
        return "\n".join(lines)
    
    def run(self) -> None:
        """
        Launch the interactive explorer.
        
        This opens a window with:
        - 3D mesh visualization
        - Slider controls for each principal component
        - Help text showing variance explained
        """
        # Create plotter
        self.plotter = pv.Plotter(
            window_size=self.window_size,
            title="Tangent PCA Shape Explorer"
        )
        self.plotter.set_background(self.COLORS['background'])
        
        # Add the mesh
        self.plotter.add_mesh(
            self.mesh,
            color=self.COLORS['mesh'],
            smooth_shading=True,
            name='shape_mesh'
        )
        
        # Add axes
        self.plotter.add_axes()
        
        # Add controls text
        self.plotter.add_text(
            self._get_controls_text(),
            position='upper_left',
            font_size=9,
            color=self.COLORS['text']
        )
        
        # Add title
        self.plotter.add_text(
            "Tangent PCA Shape Space Explorer",
            position='upper_edge',
            font_size=16,
            color=self.COLORS['text']
        )
        
        # Create sliders for each component
        slider_height = 0.90  # Start position
        slider_spacing = 0.12
        
        for i in range(self.n_components):
            var_pct = self.model.explained_variance_ratio[i] * 100
            
            self.plotter.add_slider_widget(
                callback=self._create_slider_callback(i),
                rng=(-self.sigma_range, self.sigma_range),
                value=0,
                title=f"PC{i+1} ({var_pct:.1f}%)",
                pointa=(0.7, slider_height - i * slider_spacing),
                pointb=(0.95, slider_height - i * slider_spacing),
                style='modern',
                title_opacity=0.8,
                title_color=self.COLORS['text'],
                fmt='%.2f'
            )
        
        # Add key bindings
        def on_key_r():
            self._reset_weights()
            print("[Tangent PCA Explorer] Reset to mean shape")
        
        self.plotter.add_key_event('r', on_key_r)
        self.plotter.add_key_event('R', on_key_r)
        
        # Enable camera orbit by default
        self.plotter.enable_trackball_style()
        
        # Add a nice initial camera position
        self.plotter.camera_position = 'xy'
        self.plotter.camera.azimuth = -30
        self.plotter.camera.elevation = 20
        self.plotter.reset_camera()
        
        # Show the plot
        print("[Tangent PCA Explorer] Starting interactive viewer...")
        print("[Tangent PCA Explorer] Use sliders on the right to explore shape variations")
        print("[Tangent PCA Explorer] Press 'R' to reset, 'Q' to quit")
        
        self.plotter.show()


# =============================================================================
# Convenience Function
# =============================================================================

def explore_tangent_pca(model_dir: str,
                        template_path: str,
                        n_components: int = 5,
                        sigma_range: float = 3.0) -> None:
    """
    Quick launch for Tangent PCA exploration.
    
    Args:
        model_dir: Path to saved Tangent PCA model
        template_path: Path to template OBJ file
        n_components: Number of components to control
        sigma_range: Range for sliders (±sigma)
    """
    explorer = TangentPCAExplorer(
        model_dir=model_dir,
        template_path=template_path,
        n_components=n_components,
        sigma_range=sigma_range
    )
    explorer.run()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive Tangent PCA Shape Explorer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic exploration with default settings
  python tangent_pca_explorer.py --model model/tangent_pca --template data/training/L_Femur_11.obj
  
  # Explore with 10 components and wider sigma range
  python tangent_pca_explorer.py --model model/tangent_pca --template data/training/L_Femur_11.obj --components 10 --sigma 5.0
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to directory containing saved Tangent PCA model'
    )
    parser.add_argument(
        '--template', '-t',
        required=True,
        help='Path to template OBJ file for mesh connectivity'
    )
    parser.add_argument(
        '--components', '-c',
        type=int,
        default=5,
        help='Number of principal components to control (default: 5)'
    )
    parser.add_argument(
        '--sigma', '-s',
        type=float,
        default=3.0,
        help='Sigma range for sliders (default: 3.0)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1400,
        help='Window width (default: 1400)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=900,
        help='Window height (default: 900)'
    )
    
    args = parser.parse_args()
    
    explorer = TangentPCAExplorer(
        model_dir=args.model,
        template_path=args.template,
        n_components=args.components,
        sigma_range=args.sigma,
        window_size=(args.width, args.height)
    )
    
    explorer.run()
