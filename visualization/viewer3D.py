#!/usr/bin/env python3
"""3D Viewer Module
This module provides a class to visualize 3D .obj files using PyVista.
It includes functionality to load the mesh, set up the visualization scene,
and run the interactive viewer."""

import os
import pyvista as pv
import numpy as np

class Viewer3D:
    """
    Main class to handle 3D visualization of .obj files.
    """
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.mesh = None
        self.plotter = None

        # Validation
        if not os.path.exists(self.obj_path):
            raise FileNotFoundError(f"3D file not found at: {self.obj_path}")

    def load_mesh(self):
        """
        Loads the .obj file into memory and stores the base topology.
        """
        print(f"[Info] Loading mesh from: {self.obj_path}...")

        self.mesh = pv.read(self.obj_path)

        print(f"[Success] Mesh loaded. Vertices: {self.mesh.n_points}, Faces: {self.mesh.n_cells}")

    def setup_scene(self,
                    window_size,
                    title_window,
                    color_object,
                    smooth_shading,
                    show_edges,
                    show_grid,
                    show_axes):
        """
        Configures the PyVista plotter, camera, and lighting.
        """
        # Create a window
        self.plotter = pv.Plotter(window_size=window_size, title=title_window)

        # Add the mesh to the scene
        # smooth_shading=True makes it look like bone (smooth by interpolation), not low-poly (we see the triangles)
        # show_edges=True draws the edges of the triangles
        self.plotter.add_mesh(self.mesh, color=color_object, smooth_shading=smooth_shading, show_edges=show_edges)

        # Add spatial reference
        if show_axes:
            self.plotter.add_axes()
        if show_grid:
            self.plotter.show_grid()

    def run(self,
            window_size=(1200, 800),
            title_window="Viewer3D",
            color_object="beige",
            smooth_shading=True,
            show_edges=False,
            show_grid=False,
            show_axes=True):
        """
        Starts the visualization loop.
        """
        self.load_mesh()
        self.setup_scene(window_size,
                         title_window,
                         color_object,
                         smooth_shading,
                         show_edges,
                         show_grid,
                         show_axes)
        
        print("[Info] Starting visualization window...")
        print("[Tip] Press 'q' to close the window.")
        self.plotter.show()
