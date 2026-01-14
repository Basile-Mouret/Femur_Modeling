#!/usr/bin/env python3
"""
PCA Reconstruction Analysis Tool

Analyzes reconstruction quality of PCA shape models by computing
errors and generating comparison visualizations.

Author: Femur Modeling Project
Date: 2026
"""

import os
import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from pca_visualizer import (
    load_pca_model,
    load_template_mesh,
    create_mesh_from_shape,
    shape_to_points,
    PCAModel
)


def load_shape_from_obj(obj_path: str, standardize: bool = True) -> np.ndarray:
    """
    Load a shape vector from an OBJ file in the format expected by the PCA model.
    
    The PCA model uses stacked format [x1,x2,...,xN, y1,y2,...,yN, z1,z2,...,zN]
    with standardization factors applied.
    
    Args:
        obj_path: Path to OBJ file
        standardize: Whether to apply standardization (divide by factors)
        
    Returns:
        Flattened shape vector in stacked format [all_X, all_Y, all_Z]
    """
    from pca_visualizer import STANDARDIZATION_FACTORS
    
    mesh = pv.read(obj_path)
    points = np.array(mesh.points)  # Shape: (N, 3)
    
    # Extract coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    
    # Apply standardization (same as C++ Femur::getCoordsVect)
    if standardize:
        x_coords = x_coords / STANDARDIZATION_FACTORS['x']
        y_coords = y_coords / STANDARDIZATION_FACTORS['y']
        z_coords = z_coords / STANDARDIZATION_FACTORS['z']
    
    # Stack in the format expected by PCA: [all_X, all_Y, all_Z]
    return np.concatenate([x_coords, y_coords, z_coords])


def transform_shape(model: PCAModel, shape: np.ndarray, n_components: int = None) -> np.ndarray:
    """
    Transform a shape to PCA coefficients.
    
    Args:
        model: PCA model
        shape: Shape vector
        n_components: Number of components to use
        
    Returns:
        PCA coefficients
    """
    n_components = n_components or model.n_components
    centered = shape - model.mean
    coefficients = np.zeros(n_components)
    
    for k in range(n_components):
        coefficients[k] = np.dot(centered, model.components[:, k])
    
    return coefficients


def inverse_transform(model: PCAModel, coefficients: np.ndarray) -> np.ndarray:
    """
    Reconstruct a shape from PCA coefficients.
    
    Args:
        model: PCA model
        coefficients: PCA coefficients
        
    Returns:
        Reconstructed shape vector
    """
    n_comp = len(coefficients)
    shape = model.mean.copy()
    
    for k in range(n_comp):
        shape += coefficients[k] * model.components[:, k]
    
    return shape


def compute_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """
    Compute various reconstruction error metrics.
    
    Both input vectors should be in stacked format [all_X, all_Y, all_Z].
    
    Args:
        original: Original shape vector (stacked format)
        reconstructed: Reconstructed shape vector (stacked format)
        
    Returns:
        Dictionary with error metrics
    """
    diff = original - reconstructed
    
    # Per-vertex errors (vectors are in stacked format [X,X,...,Y,Y,...,Z,Z,...])
    n_vertices = len(original) // 3
    
    # Extract X, Y, Z blocks
    orig_x = original[0:n_vertices]
    orig_y = original[n_vertices:2*n_vertices]
    orig_z = original[2*n_vertices:3*n_vertices]
    rec_x = reconstructed[0:n_vertices]
    rec_y = reconstructed[n_vertices:2*n_vertices]
    rec_z = reconstructed[2*n_vertices:3*n_vertices]
    
    # Stack into (N, 3) for distance computation
    original_pts = np.column_stack([orig_x, orig_y, orig_z])
    reconstructed_pts = np.column_stack([rec_x, rec_y, rec_z])
    
    vertex_distances = np.linalg.norm(original_pts - reconstructed_pts, axis=1)
    
    return {
        'mse': np.mean(diff ** 2),
        'rmse': np.sqrt(np.mean(diff ** 2)),
        'mae': np.mean(np.abs(diff)),
        'max_error': np.max(np.abs(diff)),
        'mean_vertex_distance': np.mean(vertex_distances),
        'max_vertex_distance': np.max(vertex_distances),
        'median_vertex_distance': np.median(vertex_distances),
        'vertex_distances': vertex_distances
    }


class ReconstructionAnalyzer:
    """
    Analyzes PCA reconstruction quality with visualizations.
    
    Example:
        >>> analyzer = ReconstructionAnalyzer('bin/pca_model.bin', 'data/training/L_Femur_11.obj')
        >>> analyzer.analyze_shape('data/validation/R_Femur_22.obj')
        >>> analyzer.plot_error_by_components('data/validation/R_Femur_22.obj')
    """
    
    def __init__(self, model_path: str, template_path: str):
        """
        Initialize the analyzer.
        
        Args:
            model_path: Path to PCA model
            template_path: Path to template mesh
        """
        print("[Analyzer] Loading PCA model...")
        self.model = load_pca_model(model_path)
        self.template = load_template_mesh(template_path)
        self.n_vertices = self.model.n_dimensions // 3
        
        print(f"[Analyzer] Model loaded: {self.model.n_components} components")
    
    def reconstruct_shape(self, shape: np.ndarray, n_components: int = None) -> np.ndarray:
        """Reconstruct a shape using n_components."""
        n_components = n_components or self.model.n_components
        coeffs = transform_shape(self.model, shape, n_components)
        return inverse_transform(self.model, coeffs)
    
    def analyze_shape(self, 
                      obj_path: str,
                      component_counts: List[int] = None) -> dict:
        """
        Analyze reconstruction quality for a shape.
        
        Args:
            obj_path: Path to OBJ file
            component_counts: List of component counts to test
            
        Returns:
            Dictionary with analysis results
        """
        # Default component counts
        if component_counts is None:
            component_counts = [1, 2, 3, 5, 7, 10, 15, 20, self.model.n_components]
            component_counts = [k for k in component_counts if k <= self.model.n_components]
        
        # Load shape
        original = load_shape_from_obj(obj_path)
        
        if len(original) != self.model.n_dimensions:
            raise ValueError(
                f"Shape dimension mismatch: expected {self.model.n_dimensions}, "
                f"got {len(original)}"
            )
        
        print(f"[Analyzer] Analyzing: {os.path.basename(obj_path)}")
        
        results = {
            'file': obj_path,
            'original': original,
            'errors_by_k': {}
        }
        
        for k in component_counts:
            reconstructed = self.reconstruct_shape(original, k)
            errors = compute_reconstruction_error(original, reconstructed)
            results['errors_by_k'][k] = errors
            
            print(f"  K={k:3d}: RMSE={errors['rmse']:.6f}, "
                  f"Mean vertex dist={errors['mean_vertex_distance']:.4f}mm")
        
        return results
    
    def plot_error_by_components(self,
                                  obj_path: str,
                                  figsize: Tuple[int, int] = (12, 5),
                                  save_path: str = None) -> None:
        """
        Plot reconstruction error vs number of components.
        
        Args:
            obj_path: Path to OBJ file
            figsize: Figure size
            save_path: If provided, save figure
        """
        # Analyze with all component counts
        k_values = list(range(1, self.model.n_components + 1))
        
        original = load_shape_from_obj(obj_path)
        
        rmse_values = []
        mean_dist_values = []
        
        for k in k_values:
            reconstructed = self.reconstruct_shape(original, k)
            errors = compute_reconstruction_error(original, reconstructed)
            rmse_values.append(errors['rmse'])
            mean_dist_values.append(errors['mean_vertex_distance'])
        
        # Compute variance explained
        variance_ratios = self.model.variances / self.model.total_variance
        cumulative_var = np.cumsum(variance_ratios) * 100
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'Reconstruction Analysis: {os.path.basename(obj_path)}', 
                    fontsize=12, fontweight='bold')
        
        # Plot 1: RMSE vs K
        axes[0].plot(k_values, rmse_values, 'o-', color='steelblue', markersize=4)
        axes[0].set_xlabel('Number of Components (K)')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Reconstruction Error (RMSE)')
        axes[0].grid(alpha=0.3)
        axes[0].set_yscale('log')
        
        # Plot 2: Mean vertex distance vs K
        axes[1].plot(k_values, mean_dist_values, 'o-', color='coral', markersize=4)
        axes[1].set_xlabel('Number of Components (K)')
        axes[1].set_ylabel('Mean Vertex Distance')
        axes[1].set_title('Mean Vertex Distance')
        axes[1].grid(alpha=0.3)
        axes[1].set_yscale('log')
        
        # Plot 3: Error vs Variance explained
        axes[2].plot(cumulative_var, mean_dist_values, 'o-', color='green', markersize=4)
        axes[2].set_xlabel('Cumulative Variance Explained (%)')
        axes[2].set_ylabel('Mean Vertex Distance')
        axes[2].set_title('Error vs Variance Explained')
        axes[2].grid(alpha=0.3)
        axes[2].axvline(x=95, color='red', linestyle='--', alpha=0.5, label='95%')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Analyzer] Figure saved: {save_path}")
        else:
            plt.show()
    
    def visualize_reconstruction(self,
                                  obj_path: str,
                                  n_components: int,
                                  window_size: Tuple[int, int] = (1600, 600),
                                  screenshot: str = None) -> None:
        """
        Visualize original vs reconstructed shape with error heatmap.
        
        Args:
            obj_path: Path to OBJ file
            n_components: Number of components for reconstruction
            window_size: Window size
            screenshot: If provided, save screenshot
        """
        original = load_shape_from_obj(obj_path)
        reconstructed = self.reconstruct_shape(original, n_components)
        
        errors = compute_reconstruction_error(original, reconstructed)
        vertex_distances = errors['vertex_distances']
        
        # Create meshes
        original_mesh = create_mesh_from_shape(original, self.template)
        reconstructed_mesh = create_mesh_from_shape(reconstructed, self.template)
        
        # Create error visualization mesh
        error_mesh = original_mesh.copy()
        error_mesh['distance'] = vertex_distances
        
        # Setup plotter
        plotter = pv.Plotter(shape=(1, 3), window_size=window_size)
        plotter.set_background('#1a1a2e')
        
        # Original
        plotter.subplot(0, 0)
        plotter.add_mesh(original_mesh, color='#E8D4B8', smooth_shading=True)
        plotter.add_text('Original', position='upper_edge', font_size=12, color='white')
        plotter.add_axes()
        
        # Reconstructed
        plotter.subplot(0, 1)
        plotter.add_mesh(reconstructed_mesh, color='#90EE90', smooth_shading=True)
        plotter.add_text(f'Reconstructed (K={n_components})', 
                        position='upper_edge', font_size=12, color='white')
        plotter.add_axes()
        
        # Error heatmap
        plotter.subplot(0, 2)
        plotter.add_mesh(
            error_mesh, 
            scalars='distance',
            cmap='hot',
            smooth_shading=True,
            scalar_bar_args={
                'title': 'Vertex Distance',
                'color': 'white',
                'title_font_size': 10,
                'label_font_size': 8
            }
        )
        plotter.add_text('Error Heatmap', position='upper_edge', font_size=12, color='white')
        plotter.add_axes()
        
        # Add statistics text
        plotter.add_text(
            f"RMSE: {errors['rmse']:.4f}\n"
            f"Mean dist: {errors['mean_vertex_distance']:.4f}\n"
            f"Max dist: {errors['max_vertex_distance']:.4f}",
            position='lower_left',
            font_size=9,
            color='white',
            viewport=True
        )
        
        if screenshot:
            plotter.show(screenshot=screenshot, auto_close=True)
            print(f"[Analyzer] Screenshot saved: {screenshot}")
        else:
            plotter.show()
    
    def batch_analyze(self, 
                      obj_dir: str,
                      output_dir: str = None,
                      component_counts: List[int] = None) -> dict:
        """
        Analyze all OBJ files in a directory.
        
        Args:
            obj_dir: Directory containing OBJ files
            output_dir: Directory for output plots
            component_counts: Component counts to test
            
        Returns:
            Dictionary with all results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find OBJ files
        obj_files = sorted([
            f for f in os.listdir(obj_dir) 
            if f.endswith('.obj')
        ])
        
        print(f"[Analyzer] Found {len(obj_files)} OBJ files in {obj_dir}")
        
        all_results = {}
        
        for filename in obj_files:
            filepath = os.path.join(obj_dir, filename)
            
            try:
                results = self.analyze_shape(filepath, component_counts)
                all_results[filename] = results
                
                if output_dir:
                    # Save error plot
                    plot_path = os.path.join(output_dir, f"{filename[:-4]}_error_plot.png")
                    self.plot_error_by_components(filepath, save_path=plot_path)
                    
            except Exception as e:
                print(f"[Analyzer] Error processing {filename}: {e}")
        
        return all_results

    def visualize_multi_reconstruction(self,
                                        obj_path: str,
                                        component_list: List[int] = None,
                                        window_size: Tuple[int, int] = (1800, 900),
                                        screenshot: str = None) -> None:
        """
        Beautiful multi-panel visualization comparing original with multiple reconstructions.
        
        Shows original shape alongside reconstructions with increasing component counts,
        plus an error heatmap panel.
        
        Args:
            obj_path: Path to OBJ file to analyze
            component_list: List of component counts to compare (default: [1, 5, 10, K])
            window_size: Window dimensions
            screenshot: If provided, save screenshot to this path
        """
        from pca_visualizer import STANDARDIZATION_FACTORS
        
        # Default component counts
        if component_list is None:
            K = self.model.n_components
            component_list = [1, 5, 10, K] if K >= 10 else [1, K//2, K]
        
        # Load and reconstruct
        original = load_shape_from_obj(obj_path)
        
        n_panels = len(component_list) + 2  # original + reconstructions + error heatmap
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        
        # Setup plotter with dark theme
        pv.set_plot_theme('dark')
        plotter = pv.Plotter(
            shape=(n_rows, n_cols), 
            window_size=window_size,
            title='PCA Reconstruction Comparison'
        )
        
        # Professional dark background
        bg_color = '#0d1117'
        
        # Color scheme
        original_color = '#E8D4B8'       # Warm bone color for original
        recon_colors = ['#7DD3FC', '#34D399', '#A78BFA', '#F472B6']  # Soft blues/greens/purples
        
        shape_name = os.path.basename(obj_path)
        
        # Panel 0: Original shape
        plotter.subplot(0, 0)
        plotter.set_background(bg_color)
        original_mesh = create_mesh_from_shape(original, self.template)
        plotter.add_mesh(
            original_mesh, 
            color=original_color, 
            smooth_shading=True,
            specular=0.3,
            specular_power=15
        )
        plotter.add_text(
            'Original', 
            position='upper_edge', 
            font_size=14, 
            color='white',
            font='arial'
        )
        plotter.add_axes(color='gray')
        
        # Panels 1 to N-1: Reconstructions with increasing K
        best_recon = None
        best_k = 0
        
        for i, k in enumerate(component_list):
            row = (i + 1) // n_cols
            col = (i + 1) % n_cols
            
            plotter.subplot(row, col)
            plotter.set_background(bg_color)
            
            reconstructed = self.reconstruct_shape(original, k)
            errors = compute_reconstruction_error(original, reconstructed)
            
            recon_mesh = create_mesh_from_shape(reconstructed, self.template)
            color = recon_colors[i % len(recon_colors)]
            
            plotter.add_mesh(
                recon_mesh, 
                color=color, 
                smooth_shading=True,
                specular=0.3,
                specular_power=15
            )
            
            # Calculate variance explained
            var_explained = self.model.variances[:k].sum() / self.model.total_variance * 100
            
            plotter.add_text(
                f'K={k} ({var_explained:.1f}% var)', 
                position='upper_edge', 
                font_size=12, 
                color='white',
                font='arial'
            )
            
            # Error stats at bottom
            plotter.add_text(
                f"RMSE: {errors['rmse']:.4f}\nMean: {errors['mean_vertex_distance']:.3f}",
                position=(0.02, 0.02),
                font_size=9,
                color='#888888',
                viewport=True
            )
            
            plotter.add_axes(color='gray')
            
            # Keep track of best reconstruction for heatmap
            if k == max(component_list):
                best_recon = reconstructed
                best_k = k
        
        # Last panel: Error heatmap on best reconstruction
        panel_idx = len(component_list) + 1
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        
        plotter.subplot(row, col)
        plotter.set_background(bg_color)
        
        errors = compute_reconstruction_error(original, best_recon)
        vertex_distances = errors['vertex_distances']
        
        # De-standardize distances for meaningful mm values
        # Average standardization factor for rough mm estimate
        avg_factor = np.mean([STANDARDIZATION_FACTORS['x'], 
                             STANDARDIZATION_FACTORS['y'], 
                             STANDARDIZATION_FACTORS['z']])
        vertex_distances_mm = vertex_distances * avg_factor
        
        error_mesh = create_mesh_from_shape(original, self.template)
        error_mesh['error_mm'] = vertex_distances_mm
        
        plotter.add_mesh(
            error_mesh, 
            scalars='error_mm',
            cmap='coolwarm',
            smooth_shading=True,
            clim=[0, np.percentile(vertex_distances_mm, 95)],  # Clip outliers
            scalar_bar_args={
                'title': 'Error (mm)',
                'color': 'white',
                'title_font_size': 11,
                'label_font_size': 9,
                'n_labels': 5,
                'position_x': 0.85,
                'position_y': 0.3,
                'width': 0.1,
                'height': 0.4
            }
        )
        
        plotter.add_text(
            f'Error Heatmap (K={best_k})', 
            position='upper_edge', 
            font_size=12, 
            color='white',
            font='arial'
        )
        
        plotter.add_text(
            f"Max: {vertex_distances_mm.max():.2f}mm\n"
            f"95%: {np.percentile(vertex_distances_mm, 95):.2f}mm\n"
            f"Mean: {vertex_distances_mm.mean():.2f}mm",
            position=(0.02, 0.02),
            font_size=9,
            color='#888888',
            viewport=True
        )
        
        plotter.add_axes(color='gray')
        
        # Link all views for synchronized rotation
        plotter.link_views()
        
        # Add main title
        plotter.add_text(
            f"Reconstruction Analysis: {shape_name}",
            position=(0.35, 0.97),
            font_size=16,
            color='white',
            viewport=True
        )
        
        print(f"\n{'='*60}")
        print(f"  Reconstruction Comparison: {shape_name}")
        print(f"{'='*60}")
        print(f"  Components tested: {component_list}")
        print(f"  Final error (K={best_k}): {errors['mean_vertex_distance']*avg_factor:.2f}mm mean")
        print(f"{'='*60}\n")
        
        if screenshot:
            plotter.show(screenshot=screenshot, auto_close=True)
            print(f"[Analyzer] Screenshot saved: {screenshot}")
        else:
            plotter.show()
        
        return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PCA Reconstruction Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single shape
  python reconstruction_analysis.py --model ../bin/pca_femur_model.bin \\
      --template ../data/training/L_Femur_11.obj \\
      --shape ../data/validation/R_Femur_22.obj
  
  # Batch analyze all shapes in a directory
  python reconstruction_analysis.py --model ../bin/pca_femur_model.bin \\
      --template ../data/training/L_Femur_11.obj \\
      --batch ../data/validation/ --output results/
  
  # Visualize reconstruction with error heatmap
  python reconstruction_analysis.py --model ../bin/pca_femur_model.bin \\
      --template ../data/training/L_Femur_11.obj \\
      --shape ../data/validation/R_Femur_22.obj \\
      --visualize --components 10
        """
    )
    
    parser.add_argument('--model', '-m', required=True, help='Path to PCA model')
    parser.add_argument('--template', '-t', required=True, help='Path to template OBJ')
    parser.add_argument('--shape', '-s', help='Path to shape OBJ to analyze')
    parser.add_argument('--batch', '-b', help='Directory of OBJ files to batch analyze')
    parser.add_argument('--output', '-o', help='Output directory for plots')
    parser.add_argument('--visualize', '-v', action='store_true', help='Show 3D visualization')
    parser.add_argument('--components', '-k', type=int, default=10, 
                       help='Number of components for visualization')
    
    args = parser.parse_args()
    
    analyzer = ReconstructionAnalyzer(args.model, args.template)
    
    if args.shape:
        if args.visualize:
            analyzer.visualize_reconstruction(args.shape, args.components)
        else:
            results = analyzer.analyze_shape(args.shape)
            analyzer.plot_error_by_components(
                args.shape, 
                save_path=os.path.join(args.output, 'error_plot.png') if args.output else None
            )
    
    if args.batch:
        analyzer.batch_analyze(args.batch, args.output)
    
    if not args.shape and not args.batch:
        parser.print_help()
