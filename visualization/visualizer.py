#!/usr/bin/env python3
import os

import pyvista as pv
import numpy as np

class FemurViewer:
    """
    Main class to handle 3D femur visualization.
    Designed to be extended with Neural Network decoding logic later.
    """
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.mesh = None
        self.plotter = None
        self.base_points = None # Will store original vertex positions

        # Validation
        if not os.path.exists(self.obj_path):
            raise FileNotFoundError(f"3D file not found at: {self.obj_path}")

    def load_mesh(self):
        """
        Loads the .obj file into memory and stores the base topology.
        """
        print(f"[Info] Loading mesh from: {self.obj_path}...")
        self.mesh = pv.read(self.obj_path)
        
        # Store a copy of the original points.
        # CRITICAL FOR LATER: When we implement the neural network, 
        # we will always calculate: new_shape = base_points + deformation
        self.base_points = self.mesh.points.copy()
        
        print(f"[Success] Mesh loaded. Vertices: {self.mesh.n_points}, Faces: {self.mesh.n_cells}")

    def setup_scene(self):
        """
        Configures the PyVista plotter, camera, and lighting.
        """
        # Create a window
        self.plotter = pv.Plotter(window_size=[1200, 800], title="Femur Visualization - Team 4")
        
        # Add the mesh to the scene
        # smooth_shading=True makes it look like bone, not low-poly
        self.plotter.add_mesh(self.mesh, color="beige", smooth_shading=True, show_edges=False)
        
        # Add spatial reference
        self.plotter.add_axes()
        self.plotter.show_grid()
        
        # Placeholder for future UI elements
        self.plotter.add_text("Femur visualization", position='upper_left', font_size=10)

    def run(self):
        """
        Starts the visualization loop.
        """
        self.load_mesh()
        self.setup_scene()
        
        print("[Info] Starting visualization window...")
        print("[Tip] Press 'q' to close the window.")
        self.plotter.show()

if __name__ == "__main__":
    path = "../data/validation/"
    femur = "R_Femur_22_DECIM.obj.FINAL.obj"

    try:
        app = FemurViewer(os.path.join(path, femur))
        app.run()
    except Exception as e:
        print(f"Application Error: {e}")