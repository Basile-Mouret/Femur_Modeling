#!/usr/bin/env python3
"""
3D Latent Space Viewer

Interactive 3D visualization of latent space projections.
Uses PCA to reduce 10D latent vectors to 3D for visualization.
Supports camera rotation, zoom, and pan.
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import sys
from sklearn.decomposition import PCA

# Paths
script_dir = Path(__file__).parent.absolute()
data_file = script_dir / "latent_projections.npz"


def load_projections():
    """Load latent projections from saved file."""
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please run 'project_training_femurs.py' first.")
        sys.exit(1)
    
    data = np.load(data_file, allow_pickle=True)
    latents = data['latents']
    femur_names = data['femur_names']
    return latents, femur_names


def reduce_to_3d(latents, method='pca', center=True, global_mean=None):
    """
    Reduce high-dimensional latent space to 3D for visualization.
    
    Args:
        latents: Array of shape (N, latent_dim)
        method: 'pca' or 'first3' (use first 3 dimensions directly)
        center: If True, center the data around 0 for better visualization
        global_mean: Pre-computed global mean for consistent centering
        
    Returns:
        points_3d: Array of shape (N, 3)
        explained_variance: Variance explained by each component (for PCA)
        center_offset: The offset used to center the data (for reference)
    """
    if method == 'first3':
        points_3d = latents[:, :3].copy()
        if center:
            if global_mean is not None:
                center_offset = global_mean[:3]
            else:
                center_offset = np.mean(points_3d, axis=0)
            points_3d = points_3d - center_offset
            # Scale to make variations more visible (multiply by 1000)
            scale_factor = 1000.0
            points_3d = points_3d * scale_factor
            print(f"Data centered around: {center_offset}")
            print(f"Scaled by {scale_factor}x for visibility")
        else:
            center_offset = np.zeros(3)
        return points_3d, None, center_offset
    
    elif method == 'pca':
        pca = PCA(n_components=3)
        points_3d = pca.fit_transform(latents)
        # PCA already centers the data
        return points_3d, pca.explained_variance_ratio_, np.zeros(3)
    
    else:
        raise ValueError(f"Unknown method: {method}")


class LatentSpace3DViewer:
    """Interactive 3D viewer for latent space."""
    
    def __init__(self, latents, femur_names, method='pca', global_mean=None):
        """
        Initialize the viewer.
        
        Args:
            latents: Array of latent vectors (N, latent_dim)
            femur_names: Array of femur names
            method: Dimensionality reduction method ('pca' or 'first3')
            global_mean: Pre-computed global mean for consistent centering
        """
        self.latents = latents
        self.femur_names = femur_names
        self.method = method
        
        # Reduce to 3D
        print(f"Reducing {latents.shape[1]}D latent space to 3D using {method}...")
        self.points_3d, self.explained_var, self.center_offset = reduce_to_3d(latents, method, global_mean=global_mean)
        
        if self.explained_var is not None:
            print(f"Explained variance ratio: {self.explained_var}")
            print(f"Total explained: {sum(self.explained_var):.2%}")
        
        # Separate left and right femurs
        self.left_mask = np.array(['L_' in str(name) for name in femur_names])
        self.right_mask = ~self.left_mask
        
        # Create short names for labels
        self.short_names = [
            str(name).replace('_DECIM.obj.FINAL', '')
                     .replace('L_Femur_', 'L')
                     .replace('R_Femur_', 'R')
            for name in femur_names
        ]
        
        # Setup plotter
        self._setup_plotter()
    
    def _setup_plotter(self):
        """Setup the PyVista plotter."""
        self.plotter = pv.Plotter(title="Latent Space 3D Viewer")
        
        # Add left femurs (blue spheres)
        if np.any(self.left_mask):
            left_points = self.points_3d[self.left_mask]
            left_cloud = pv.PolyData(left_points)
            self.plotter.add_mesh(
                left_cloud,
                color='blue',
                point_size=20,
                render_points_as_spheres=True,
                label='Left Femurs'
            )
            # Add labels for left femurs
            left_names = [n for n, m in zip(self.short_names, self.left_mask) if m]
            for i, (point, name) in enumerate(zip(left_points, left_names)):
                self.plotter.add_point_labels(
                    [point], [name],
                    font_size=12,
                    point_color='blue',
                    point_size=0,
                    shape=None,
                    fill_shape=False,
                    margin=5,
                    always_visible=True
                )
        
        # Add right femurs (red spheres)
        if np.any(self.right_mask):
            right_points = self.points_3d[self.right_mask]
            right_cloud = pv.PolyData(right_points)
            self.plotter.add_mesh(
                right_cloud,
                color='red',
                point_size=20,
                render_points_as_spheres=True,
                label='Right Femurs'
            )
            # Add labels for right femurs
            right_names = [n for n, m in zip(self.short_names, self.right_mask) if m]
            for i, (point, name) in enumerate(zip(right_points, right_names)):
                self.plotter.add_point_labels(
                    [point], [name],
                    font_size=12,
                    point_color='red',
                    point_size=0,
                    shape=None,
                    fill_shape=False,
                    margin=5,
                    always_visible=True
                )
        
        # Add mean point (green)
        mean_point = np.mean(self.points_3d, axis=0, keepdims=True)
        mean_cloud = pv.PolyData(mean_point)
        self.plotter.add_mesh(
            mean_cloud,
            color='green',
            point_size=25,
            render_points_as_spheres=True,
            label='Mean'
        )
        
        # Add axes
        self.plotter.add_axes()
        
        # Add grid/floor
        self.plotter.show_grid(
            xtitle='PC0' if self.method == 'pca' else 'z0',
            ytitle='PC1' if self.method == 'pca' else 'z1',
            ztitle='PC2' if self.method == 'pca' else 'z2',
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True,
            color='gray',
            font_size=10,
            bold=False
        )
        
        # Add legend
        self.plotter.add_legend()
        
        # Set axis labels
        if self.method == 'pca':
            xlabel = f"PC1 ({self.explained_var[0]:.1%})"
            ylabel = f"PC2 ({self.explained_var[1]:.1%})"
            zlabel = f"PC3 ({self.explained_var[2]:.1%})"
        else:
            xlabel, ylabel, zlabel = "z0", "z1", "z2"
        
        # Add title with info
        title = f"Latent Space (N={len(self.femur_names)}, dim={self.latents.shape[1]}→3D via {self.method.upper()})"
        self.plotter.add_title(title, font_size=12)
        
        # Add text instructions
        self.plotter.add_text(
            "Controls:\n"
            "  Left-click + drag: Rotate\n"
            "  Scroll: Zoom\n"
            "  Middle-click: Pan\n"
            "  R: Reset camera\n"
            "  Q: Quit",
            position='lower_left',
            font_size=9,
            color='gray'
        )
        
        # Set initial camera position
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()
    
    def show(self):
        """Display the interactive viewer."""
        print("\n=== 3D Latent Space Viewer ===")
        print("Controls:")
        print("  • Left-click + drag: Rotate camera")
        print("  • Scroll wheel: Zoom in/out")
        print("  • Middle-click + drag: Pan")
        print("  • R: Reset camera view")
        print("  • Q: Quit")
        print()
        self.plotter.show()


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='3D Latent Space Viewer')
    parser.add_argument('--method', choices=['pca', 'first3'], default='first3',
                       help='Dimensionality reduction method (default: first3)')
    parser.add_argument('-n', '--count', type=int, default=1,
                       help='Number of femurs to display (default: 1, use -1 for all)')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting index (default: 0)')
    args = parser.parse_args()
    
    # Load data
    print("Loading latent projections...")
    all_latents, all_femur_names = load_projections()
    print(f"Loaded {len(all_femur_names)} femurs with {all_latents.shape[1]}-dimensional latent space")
    
    # Compute global mean for consistent centering
    global_mean = np.mean(all_latents, axis=0)
    print(f"Global mean (z0, z1, z2): [{global_mean[0]:.6f}, {global_mean[1]:.6f}, {global_mean[2]:.6f}]")
    
    # Filter to selected femurs
    if args.count == -1:
        # Show all
        print(f"Displaying all {len(all_femur_names)} femurs")
        latents = all_latents
        femur_names = all_femur_names
    else:
        end_idx = min(args.start + args.count, len(all_femur_names))
        if args.start >= len(all_femur_names):
            print(f"Error: start index {args.start} out of range (0-{len(all_femur_names)-1})")
            sys.exit(1)
        print(f"Displaying femurs {args.start} to {end_idx-1} ({end_idx - args.start} femurs)")
        latents = all_latents[args.start:end_idx]
        femur_names = all_femur_names[args.start:end_idx]
    
    # Create and show viewer
    viewer = LatentSpace3DViewer(latents, femur_names, method=args.method, global_mean=global_mean)
    viewer.show()


if __name__ == "__main__":
    main()
