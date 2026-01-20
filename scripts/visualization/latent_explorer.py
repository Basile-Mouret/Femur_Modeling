#!/usr/bin/env python3
"""
Interactive Latent Space Explorer for Femur Neural Network

This script provides a 3D viewer with 10 sliders that control
the latent space of the autoencoder. Moving sliders triggers
the C++ decoder and updates the mesh in real-time.
"""

import numpy as np
import pyvista as pv
import os
import sys

# Add visualization directory to path for femur_rdn module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Add 'lib' directory to sys.path for femur_rdn import
lib_dir = os.path.abspath(os.path.join(script_dir, 'lib'))
sys.path.insert(0, lib_dir)

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


class LatentExplorer:
    """Interactive explorer for the neural network's latent space."""
    
    def __init__(self, model_path: str, faces_obj_path: str, baseline_femur_path: str = None):
        """
        Initialize the explorer.
        
        Args:
            model_path: Path to the neural network .bin file
            faces_obj_path: Path to OBJ file for face connectivity
            baseline_femur_path: Path to a real femur OBJ for baseline latent encoding
        """
        # Initialize decoder
        print(f"Loading model from: {model_path}")
        femur_rdn.init_decoder(model_path)
        
        self.latent_size = femur_rdn.get_latent_size()
        self.num_points = femur_rdn.get_num_points()
        
        print(f"Latent space size: {self.latent_size}")
        print(f"Number of vertices: {self.num_points}")
        
        # Load faces from reference OBJ (for mesh connectivity)
        print(f"Loading faces from: {faces_obj_path}")
        self.faces = load_faces_from_obj(faces_obj_path)
        
        # Load real training femur and encode to get baseline latent values
        baseline_path = baseline_femur_path if baseline_femur_path else faces_obj_path
        print(f"Encoding baseline femur from: {baseline_path}")
        ref_vertices = load_vertices_from_obj(baseline_path)
        self.baseline_latent = np.array(femur_rdn.encode(ref_vertices), dtype=np.float64)
        print(f"Baseline latent values: {self.baseline_latent}")
        
        # Initialize latent vector to baseline (reference femur)
        self.latent_values = self.baseline_latent.copy()
        
        # Create initial mesh
        self._update_mesh_data()
        self.mesh = pv.PolyData(self.vertices, self.faces)
        
        # Setup plotter
        self.plotter = pv.Plotter(title="Femur Latent Space Explorer")
        self.actor = self.plotter.add_mesh(
            self.mesh, 
            color='white',
            show_edges=False,
            smooth_shading=True,
            specular=0.5
        )
        
        # Add sliders
        self._add_sliders()
        
        # Add coordinate axes
        self.plotter.add_axes()
        
        # Set initial camera
        self.plotter.camera_position = 'xy'
        self.plotter.reset_camera()
    
    def _update_mesh_data(self):
        """Call C++ decoder and update vertex positions."""
        # Call decoder with current latent values
        # decode() returns numpy array of shape (n_points, 3)
        self.vertices = femur_rdn.decode(self.latent_values.tolist())
    
    def _on_slider_change(self, idx: int):
        """Factory function to create slider callbacks."""
        def callback(value):
            self.latent_values[idx] = value
            self._update_mesh_data()
            self.mesh.points = self.vertices
            # Only call update if the interactor is initialized
            try:
                if hasattr(self.plotter, 'iren') and hasattr(self.plotter.iren, 'GetInitialized'):
                    if self.plotter.iren.GetInitialized():
                        self.plotter.update()
            except Exception:
                pass  # Silently ignore update errors
        return callback
    
    def _add_sliders(self):
        """Add sliders for each latent dimension, range [0, 1]."""
        slider_height = 0.04
        spacing = 0.005
        start_y = 0.95

        for i in range(self.latent_size):
            y_pos = start_y - i * (slider_height + spacing)
            self.plotter.add_slider_widget(
                callback=self._on_slider_change(i),
                rng=[0.0, 1.0],
                value=np.clip(self.baseline_latent[i], 0.0, 1.0),
                title=f"z{i}",
                pointa=(0.02, y_pos - slider_height),
                pointb=(0.18, y_pos - slider_height),
                style='modern',
            )
    
    def show(self):
        """Display the interactive viewer."""
        print("\n=== Latent Space Explorer ===")
        print("Use the sliders to explore the latent space.")
        print("Each slider controls one latent dimension (z0-z9).")
        print("Close the window to exit.\n")
        self.plotter.show()


def main():
    # Paths
    project_root = os.path.dirname(script_dir) + "/../"
    if len(sys.argv) != 3:
        print("Usage: ./latent_explorer.py <path_to_neural_network.bin> <path_to_base_femur.obj>")
        sys.exit(1)
    model_path = sys.argv[1]
    faces_obj_path = os.path.join(script_dir, "reconstruction_data", "base_femur_for_visu.obj")
    
    training_femur_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please ensure the trained model exists at this location.")
        sys.exit(1)
    
    if not os.path.exists(faces_obj_path):
        print(f"Error: Reference OBJ not found: {faces_obj_path}")
        print("Please ensure base_femur_for_visu.obj exists.")
        sys.exit(1)
    
    if not os.path.exists(training_femur_path):
        print(f"Error: Training femur not found: {training_femur_path}")
        print("Using faces_obj for baseline instead.")
        training_femur_path = faces_obj_path
    
    # Create and run explorer
    explorer = LatentExplorer(model_path, faces_obj_path, training_femur_path)
    explorer.show()


if __name__ == "__main__":
    main()
