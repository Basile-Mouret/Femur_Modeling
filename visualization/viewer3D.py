#!/usr/bin/env python3
"""
3D Viewer for femur .obj files.
- Point cloud (no faces) → use reference faces from base_femur_for_visu.obj
- Surface (with faces) → display surface directly
"""

import os
import numpy as np
import pyvista as pv

# Reference femur for faces (same vertex order as neural network output)
BASE_FEMUR_PATH = os.path.join(os.path.dirname(__file__), "data_visu", "base_femur_for_visu.obj")


class Viewer3D:
    def __init__(self, obj_path):
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"File not found: {obj_path}")
        self.obj_path = obj_path
        self.mesh = None
        self.plotter = None
        self._reference_faces = None

    def _is_point_cloud(self, mesh):
        """Check if mesh is a point cloud (no real faces)."""
        return mesh.n_cells == 0 or mesh.n_cells == mesh.n_points

    def _load_reference_faces(self):
        """Load faces from reference femur .obj file (cached)."""
        if self._reference_faces is not None:
            return self._reference_faces
        
        faces = []
        with open(BASE_FEMUR_PATH, 'r') as f:
            for line in f:
                if line.startswith('f '):
                    parts = line.strip().split()[1:]
                    # Handle "f 1 2 3" or "f 1/1/1 2/2/2 3/3/3"
                    indices = [int(p.split('/')[0]) - 1 for p in parts]
                    faces.append(indices)
        
        self._reference_faces = faces
        return faces

    def _apply_reference_faces(self, points):
        """Create mesh from points using reference faces."""
        faces = self._load_reference_faces()
        pv_faces = np.hstack([[len(f)] + f for f in faces])
        return pv.PolyData(points, pv_faces)

    def run(self,
            title="Viewer3D",
            color="beige",
            smooth_shading=True,
            show_edges=False,
            show_axes=True,
            show_grid=False,
            window_size=(1200, 800)):
        """
        Launch viewer.
        
        - Point cloud → use reference faces from base_femur_for_visu.obj
        - Surface mesh → display surface directly
        """
        print(f"[Info] Loading: {self.obj_path}")
        self.mesh = pv.read(self.obj_path)
        
        if self._is_point_cloud(self.mesh):
            print(f"[Info] Point cloud: {self.mesh.n_points} points")
            print(f"[Info] Applying reference faces from: {BASE_FEMUR_PATH}")
            self.mesh = self._apply_reference_faces(np.asarray(self.mesh.points))
        
        print(f"[Info] Surface: {self.mesh.n_points} vertices, {self.mesh.n_cells} faces")
        
        # Setup scene
        self.plotter = pv.Plotter(window_size=window_size, title=title)
        self.plotter.add_mesh(self.mesh, color=color, 
                              smooth_shading=smooth_shading, show_edges=show_edges)
        
        if show_axes:
            self.plotter.add_axes()
        if show_grid:
            self.plotter.show_grid()
        
        print("[Info] Press 'q' to close")
        self.plotter.show()
