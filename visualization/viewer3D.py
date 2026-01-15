#!/usr/bin/env python3
"""
3D Viewer for .obj files.
- Point cloud (no faces) → display points OR reconstruct surface with BPA
- Surface (with faces) → display surface directly
"""

import os
import numpy as np
import pyvista as pv
import open3d as o3d


class Viewer3D:
    def __init__(self, obj_path):
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"File not found: {obj_path}")
        self.obj_path = obj_path
        self.mesh = None
        self.plotter = None
        self.has_faces = False

    def _is_point_cloud(self, mesh):
        """Check if mesh is a point cloud (no real faces)."""
        return mesh.n_cells == 0 or mesh.n_cells == mesh.n_points

    def _reconstruct_bpa(self):
        """Reconstruct surface from point cloud using Ball Pivoting Algorithm."""
        print("[Info] Reconstructing surface (BPA)...")
        
        points = np.asarray(self.mesh.points)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
        radii = [avg_dist * r for r in [1.5, 2.0, 3.0]]
        
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        pv_faces = np.hstack([[3, *f] for f in faces])
        
        print(f"[Info] Reconstructed: {len(vertices)} vertices, {len(faces)} faces")
        return pv.PolyData(vertices, pv_faces)

    def run(self,
            title="Viewer3D",
            color="beige",
            smooth_shading=True,
            show_edges=False,
            show_axes=True,
            show_grid=False,
            window_size=(1200, 800),
            reconstruct_surface=False):
        """
        Launch viewer.
        
        - Point cloud + reconstruct_surface=False → display points
        - Point cloud + reconstruct_surface=True → reconstruct and display surface
        - Surface mesh → display surface directly
        """
        print(f"[Info] Loading: {self.obj_path}")
        self.mesh = pv.read(self.obj_path)
        
        is_cloud = self._is_point_cloud(self.mesh)
        
        if is_cloud:
            print(f"[Info] Point cloud: {self.mesh.n_points} points")
            if reconstruct_surface:
                self.mesh = self._reconstruct_bpa()
                self.has_faces = True
            else:
                self.has_faces = False
        else:
            print(f"[Info] Surface: {self.mesh.n_points} vertices, {self.mesh.n_cells} faces")
            self.has_faces = True
        
        # Setup scene
        self.plotter = pv.Plotter(window_size=window_size, title=title)
        
        if self.has_faces:
            self.plotter.add_mesh(self.mesh, color=color, 
                                  smooth_shading=smooth_shading, show_edges=show_edges)
        else:
            self.plotter.add_mesh(self.mesh, color=color, style="points",
                                  point_size=5, render_points_as_spheres=True)
        
        if show_axes:
            self.plotter.add_axes()
        if show_grid:
            self.plotter.show_grid()
        
        print("[Info] Press 'q' to close")
        self.plotter.show()
