#!/usr/bin/env python3
"""
Interactive PCA Explorer for Femur Neural Network

This script provides a 3D viewer with sliders that control
the PCA components of the latent space. Moving sliders modifies
the PC coordinates, which are then converted to latent space
and passed to the C++ decoder to update the mesh in real-time.

Unlike latent_explorer.py which directly modifies latent coordinates,
this script works in PCA space and automatically handles the
conversion to latent space.
"""

import numpy as np
import pyvista as pv
import os
import sys
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add visualization directory to path for femur_rdn module
script_dir = Path(__file__).parent.absolute()
lib_dir = script_dir / 'lib'
sys.path.insert(0, str(lib_dir))

try:
    import femur_rdn
except ImportError as e:
    print("Error: Could not import femur_rdn module")
    print("Make sure to compile it first:")
    print("  cd build && cmake .. && make femur_rdn")
    print(f"Details: {e}")
    sys.exit(1)


def load_faces_from_obj(obj_path: str) -> np.ndarray:
    """
    Load face connectivity from an OBJ file.
    Returns faces in PyVista format: [n_verts, v0, v1, v2, ...]
    """
    faces_list = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()[1:]
                # Handle formats: "v", "v/vt", "v/vt/vn", "v//vn"
                indices = []
                for p in parts:
                    idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                    indices.append(idx)
                faces_list.append(indices)
    
    # Convert to PyVista format
    pv_faces = []
    for face in faces_list:
        pv_faces.append(len(face))
        pv_faces.extend(face)
    
    return np.array(pv_faces, dtype=np.int64)


def load_vertices_from_obj(obj_path: str) -> np.ndarray:
    """
    Load vertex positions from an OBJ file.
    Returns vertices as numpy array (N, 3).
    """
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v ') and not line.startswith('vn') and not line.startswith('vt'):
                parts = line.strip().split()[1:]
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(vertices, dtype=np.float32)


def load_projections(model_name: str = None) -> tuple:
    """Load latent projections from saved file."""
    projection_dir = script_dir / "latent_projection"
    
    if model_name:
        data_file = projection_dir / f"latent_projections_{Path(model_name).stem}.npz"
        if not data_file.exists():
            data_file = projection_dir / "latent_projections.npz"
    else:
        data_file = projection_dir / "latent_projections.npz"
    
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please run 'project_training_femurs.py' first to generate latent projections.")
        sys.exit(1)
    
    print(f"Loading projections from: {data_file}")
    data = np.load(data_file, allow_pickle=True)
    latents = data['latents']
    femur_names = data['femur_names']
    model_used = str(data['model_name']) if 'model_name' in data else 'unknown'
    return latents, femur_names, model_used


class LatentPCAExplorer:
    """Interactive explorer using PCA of the neural network's latent space."""
    
    def __init__(self, model_path: str, faces_obj_path: str, 
                 baseline_femur_path: str = None, n_components: int = 10,
                 start_at_mean: bool = False):
        """
        Initialize the explorer.
        
        Args:
            model_path: Path to the neural network .bin file
            faces_obj_path: Path to OBJ file for face connectivity
            baseline_femur_path: Path to a real femur OBJ for baseline latent encoding
            n_components: Number of PCA components to use (max 10)
            start_at_mean: If True, start with all PC values at 0 (mean of training data)
        """
        self.start_at_mean = start_at_mean
        
        # Compute absolute path to mean_femur.obj (in data/ folder at project root)
        project_root = script_dir.parent.parent
        mean_femur_path = project_root / 'data' / 'mean_femur.obj'
        
        # Initialize decoder with correct mean femur path
        print(f"Loading model from: {model_path}")
        print(f"Using mean femur: {mean_femur_path}")
        femur_rdn.init_decoder(str(model_path), 3, str(mean_femur_path))
        
        self.latent_size = femur_rdn.get_latent_size()
        self.num_points = femur_rdn.get_num_points()
        
        print(f"Latent space size: {self.latent_size}")
        print(f"Number of vertices: {self.num_points}")
        
        # Get the activation function to determine slider ranges
        self.activation_function = femur_rdn.get_activation_function()
        print(f"Activation function: {self.activation_function}")
        
        # Load faces from reference OBJ (for mesh connectivity)
        print(f"Loading faces from: {faces_obj_path}")
        self.faces = load_faces_from_obj(faces_obj_path)
        
        # Load mean femur vertices for heat map comparison
        print(f"Loading mean femur vertices from: {mean_femur_path}")
        self.mean_vertices = load_vertices_from_obj(str(mean_femur_path))
        
        # Load baseline femur vertices and encode to get baseline latent
        baseline_path = baseline_femur_path if baseline_femur_path else faces_obj_path
        print(f"Encoding baseline femur from: {baseline_path}")
        ref_vertices = load_vertices_from_obj(baseline_path)
        self.baseline_latent = np.array(femur_rdn.encode(ref_vertices), dtype=np.float64)
        print(f"Baseline latent values: {self.baseline_latent}")
        
        # Load latent projections and fit PCA
        model_name = Path(model_path).stem
        print(f"\nLoading latent projections for PCA fitting...")
        latents, femur_names, model_used = load_projections(model_name)
        
        # Fit PCA on the latent projections
        # Always compute 10 components, but only show n_components sliders
        self.n_sliders = min(n_components, self.latent_size, 10)
        self.n_pca_components = min(self.latent_size, 10)  # Always use 10 (or max available)
        print(f"Computing PCA with {self.n_pca_components} components")
        print(f"Displaying {self.n_sliders} sliders")
        
        # Standardize the latent space data
        self.scaler = StandardScaler()
        latents_scaled = self.scaler.fit_transform(latents)
        
        # Fit PCA with all 10 components (or max available)
        self.pca = PCA(n_components=self.n_pca_components)
        self.pca.fit(latents_scaled)
        
        # Print variance explained by displayed components
        explained_var = self.pca.explained_variance_ratio_[:self.n_sliders]
        cumulative_var = np.cumsum(explained_var)
        print(f"\nPCA Variance explained by {self.n_sliders} displayed components:")
        for i, (var, cum) in enumerate(zip(explained_var, cumulative_var)):
            print(f"  PC{i}: {var:.2%} (cumulative: {cum:.2%})")
        print(f"Total variance explained by displayed components: {cumulative_var[-1]:.2%}")
        
        # Print total variance explained by all 10 components
        total_var_all = np.sum(self.pca.explained_variance_ratio_)
        print(f"Total variance explained by all {self.n_pca_components} components: {total_var_all:.2%}")
        
        # Compute PC range for sliders based on training data
        pc_projections = self.pca.transform(latents_scaled)
        self.pc_mins = pc_projections.min(axis=0)
        self.pc_maxs = pc_projections.max(axis=0)
        
        # Add some margin to the ranges
        pc_ranges = self.pc_maxs - self.pc_mins
        self.pc_mins -= 0.2 * pc_ranges
        self.pc_maxs += 0.2 * pc_ranges
        
        print(f"\nPC slider ranges (based on training data):")
        for i in range(self.n_sliders):
            print(f"  PC{i}: [{self.pc_mins[i]:.2f}, {self.pc_maxs[i]:.2f}]")
        
        # Transform baseline latent to PC space
        baseline_scaled = self.scaler.transform(self.baseline_latent.reshape(1, -1))
        self.baseline_pca = self.pca.transform(baseline_scaled).flatten()
        print(f"\nBaseline PCA coordinates (all {self.n_pca_components}): {self.baseline_pca}")
        
        # Initialize PCA values
        if self.start_at_mean:
            # Start at center of PCA space (all zeros = mean of training data)
            self.pca_values = np.zeros(self.n_pca_components)
            print("Starting at mean (all PC values = 0)")
        else:
            # Start at baseline femur's PCA coordinates
            self.pca_values = self.baseline_pca.copy()
        
        # Create initial mesh
        self._update_mesh_data()
        self.mesh = pv.PolyData(self.vertices, self.faces)
        
        # Add signed error scalars for heat map
        self._compute_signed_error()
        self.mesh["SignedError"] = self.signed_error
        
        # Setup plotter
        self.plotter = pv.Plotter(title="Femur PCA Latent Space Explorer")
        
        # Compute initial color limits based on reasonable range
        self.clim = self._compute_clim()
        
        self.actor = self.plotter.add_mesh(
            self.mesh,
            scalars="SignedError",
            cmap="coolwarm",
            show_edges=False,
            smooth_shading=True,
            specular=0.5,
            clim=self.clim,
            scalar_bar_args={
                "title": "Distance from Mean (mm)",
                "title_font_size": 14,
                "label_font_size": 10,
                "n_labels": 5,
                "fmt": "%.2f"
            }
        )
        
        # Add sliders
        self._add_sliders()
        
        # Add coordinate axes
        self.plotter.add_axes()
        
        # Set initial camera
        self.plotter.camera_position = 'xy'
        self.plotter.reset_camera()
    
    def _pca_to_latent(self, pca_coords: np.ndarray) -> np.ndarray:
        """
        Convert PCA coordinates back to latent space.
        
        Args:
            pca_coords: Full PCA coordinates vector
            
        Returns:
            Latent space vector
        """
        # Inverse transform: PCA space -> scaled latent space
        latent_scaled = self.pca.inverse_transform(pca_coords.reshape(1, -1))
        # Inverse scale: scaled latent -> original latent
        latent = self.scaler.inverse_transform(latent_scaled)
        return latent.flatten()
    
    def _compute_signed_error(self):
        """
        Compute per-vertex signed distance from mean femur.
        Positive = vertex is farther from origin than mean
        Negative = vertex is closer to origin than mean
        """
        # Compute distance vectors from mean
        diff = self.vertices - self.mean_vertices
        # Signed error: sum of differences (gives directional information)
        self.signed_error = np.linalg.norm(diff, axis=1)
        # Make it signed based on whether we're "outside" or "inside" the mean
        # Use dot product with position to determine sign
        sign = np.sign(np.sum(diff * self.mean_vertices, axis=1))
        self.signed_error = sign * self.signed_error
    
    def _compute_clim(self):
        """Compute symmetric color limits for the heat map."""
        max_abs = max(abs(self.signed_error.min()), abs(self.signed_error.max()), 0.1)
        return [-max_abs, max_abs]
    
    def _update_mesh_data(self):
        """Convert PCA to latent, call C++ decoder and update vertex positions."""
        # Convert current PCA values to latent space
        latent_values = self._pca_to_latent(self.pca_values)
        
        # Call decoder with latent values
        self.vertices = femur_rdn.decode(latent_values.tolist())
        
        # Update signed error for heat map (only if mean_vertices is defined)
        if hasattr(self, 'mean_vertices'):
            self._compute_signed_error()
    
    def _on_slider_change(self, idx: int):
        """Factory function to create slider callbacks."""
        def callback(value):
            self.pca_values[idx] = value
            self._update_mesh_data()
            self.mesh.points = self.vertices
            # Update heat map scalars
            self.mesh["SignedError"] = self.signed_error
            # Only call update if the interactor is initialized
            try:
                if hasattr(self.plotter, 'iren') and hasattr(self.plotter.iren, 'GetInitialized'):
                    if self.plotter.iren.GetInitialized():
                        self.plotter.update()
            except Exception:
                pass  # Silently ignore update errors
        return callback
    
    def _add_sliders(self):
        """Add sliders for each displayed PCA dimension."""
        slider_height = 0.04
        spacing = 0.005
        start_y = 0.95

        for i in range(self.n_sliders):
            y_pos = start_y - i * (slider_height + spacing)
            
            # Get range for this PC
            min_val = self.pc_mins[i]
            max_val = self.pc_maxs[i]
            
            # Initial value: 0 if start_at_mean, else baseline clipped to range
            if self.start_at_mean:
                initial_value = 0.0
            else:
                initial_value = np.clip(self.baseline_pca[i], min_val, max_val)
            
            # Calculate variance explained
            var_pct = self.pca.explained_variance_ratio_[i] * 100
            
            self.plotter.add_slider_widget(
                callback=self._on_slider_change(i),
                rng=[min_val, max_val],
                value=initial_value,
                title=f"PC{i} ({var_pct:.1f}%)",
                pointa=(0.02, y_pos - slider_height),
                pointb=(0.22, y_pos - slider_height),
                style='modern',
            )
    
    def show(self):
        """Display the interactive viewer."""
        print("\n=== PCA Latent Space Explorer ===")
        print(f"PCA computed with {self.n_pca_components} components.")
        print(f"Displaying {self.n_sliders} slider(s).")
        print("Use the sliders to explore the PCA space.")
        print("Each slider controls one PC dimension (PC0-PC{}).".format(self.n_sliders - 1))
        print("The percentage shows variance explained by each component.")
        print("Non-displayed components remain at their baseline values.")
        print("\nHeat map colors:")
        print("  Blue = closer to origin than mean femur")
        print("  Red  = farther from origin than mean femur")
        print("Close the window to exit.\n")
        self.plotter.show()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Interactive PCA Explorer for Femur Latent Space',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default 10 PCs
  python latent_pca_explorer.py models/NeuralNetwork.bin mean_femur.obj

  # Use only 5 PCs
  python latent_pca_explorer.py models/NeuralNetwork.bin mean_femur.obj -n 5

  # Start at mean (white femur, all sliders at 0)
  python latent_pca_explorer.py models/NeuralNetwork.bin mean_femur.obj --start-at-mean
        """
    )
    parser.add_argument('model_path', type=str,
                       help='Path to the neural network .bin file')
    parser.add_argument('baseline_femur', type=str,
                       help='Path to baseline femur OBJ file')
    parser.add_argument('-n', '--n-components', type=int, default=10,
                       help='Number of PCA components to use (max 10, default: 10)')
    parser.add_argument('--start-at-mean', action='store_true',
                       help='Start with all PC sliders at 0 (mean of training data)')
    args = parser.parse_args()
    
    # Validate n_components
    n_components = min(max(1, args.n_components), 10)
    if n_components != args.n_components:
        print(f"Note: Adjusted n_components from {args.n_components} to {n_components}")
    
    # Paths
    project_root = script_dir.parent.parent
    model_path = args.model_path
    baseline_femur_path = args.baseline_femur
    faces_obj_path = script_dir / "reconstruction_data" / "base_femur_for_visu.obj"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please ensure the trained model exists at this location.")
        sys.exit(1)
    
    if not os.path.exists(faces_obj_path):
        print(f"Error: Reference OBJ not found: {faces_obj_path}")
        print("Please ensure base_femur_for_visu.obj exists in reconstruction_data/")
        sys.exit(1)
    
    if not os.path.exists(baseline_femur_path):
        print(f"Error: Baseline femur not found: {baseline_femur_path}")
        print("Using faces_obj for baseline instead.")
        baseline_femur_path = str(faces_obj_path)
    
    # Create and run explorer
    explorer = LatentPCAExplorer(
        model_path, 
        str(faces_obj_path), 
        baseline_femur_path,
        n_components=n_components,
        start_at_mean=args.start_at_mean
    )
    explorer.show()


if __name__ == "__main__":
    main()
