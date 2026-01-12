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
                    show_axes,
                    render_style):
        """
        Configures the PyVista plotter, camera, and lighting.
        """
        # Create a window
        self.plotter = pv.Plotter(window_size=window_size, title=title_window)

        # Add the mesh to the scene with different rendering styles
        if render_style.lower() == "points": # Render as point cloud (vertices only)
            self.plotter.add_mesh(self.mesh, 
                                 color=color_object, 
                                 style="points",
                                 point_size=5,
                                 render_points_as_spheres=True)
        else:
            # Default: render as surface with triangles
            self.plotter.add_mesh(self.mesh, 
                                 color=color_object, 
                                 smooth_shading=smooth_shading, 
                                 show_edges=show_edges)

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
            show_axes=True,
            render_style="surface"):
        """
        Starts the visualization loop.
        
        Args:
            window_size: Tuple defining the size of the window (width, height)
            title_window: Title of the visualization window
            color_object: Color of the 3D object
            smooth_shading: Bool to enable smooth shading - makes it look like bone (smooth by interpolation), not low-poly (we see the triangles)
            show_edges: Bool to display mesh edges - draws the edges of the triangles
            show_grid: Bool to display a grid in the background
            show_axes: Bool to display coordinate axes
            render_style: "surface" for triangles, "points" for point cloud
        """
        self.load_mesh()
        self.setup_scene(window_size,
                         title_window,
                         color_object,
                         smooth_shading,
                         show_edges,
                         show_grid,
                         show_axes,
                         render_style)
        
        print("[Info] Starting visualization window...")
        print("[Tip] Press 'q' to close the window.")
        self.plotter.show()
