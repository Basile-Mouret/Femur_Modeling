#!/usr/bin/env python3
"""
Compare a femur OBJ file to the mean femur OBJ file and visualize per-vertex distances:
- Blue: |distance| < mean distance
- Red: |distance| >= mean distance
"""
import os
import sys
import numpy as np
import pyvista as pv

# Helper to load vertices from OBJ file
def load_vertices(obj_path):
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) == 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

# Helper to load faces from reference OBJ file
def load_faces(obj_path):
    faces = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()[1:]
                indices = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(indices)
    return faces

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_femur_to_mean.py <mean_femur.obj> <femur.obj>")
        sys.exit(1)
    mean_path = sys.argv[1]
    femur_path = sys.argv[2]
    # Optional: reference faces
    ref_faces_path = os.path.join(os.path.dirname(__file__), 'reconstruction_data', 'base_femur_for_visu.obj')

    mean_vertices = load_vertices(mean_path)
    femur_vertices = load_vertices(femur_path)
    assert femur_vertices.shape == mean_vertices.shape, "Vertex count mismatch!"

    # Compute per-vertex Euclidean distances
    distances = np.linalg.norm(femur_vertices - mean_vertices, axis=1)
    mean_dist = np.mean(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # If all distances are zero, show the femur with standard lighting using Viewer3D
    if np.allclose(distances, 0):
        print("All distances are zero. Displaying femur with standard lighting.")
        from lib.viewer3D import Viewer3D
        viewer = Viewer3D(femur_path)
        viewer.run(title="Standard Femur", color="beige", smooth_shading=True, show_edges=False, show_axes=True, show_grid=False, window_size=(1200, 800))
        sys.exit(0)

    # Gradient color: blue (no change) to red (max change)
    if max_dist > min_dist:
        t = (distances - min_dist) / (max_dist - min_dist)
    else:
        t = np.zeros_like(distances)
    colors = np.stack([t, np.zeros_like(t), 1-t], axis=1)

    # Load faces
    faces = load_faces(ref_faces_path)
    pv_faces = np.hstack([[len(f)] + f for f in faces])
    mesh = pv.PolyData(femur_vertices, pv_faces)
    mesh["ComparisonColor"] = colors  # Per-vertex RGB gradient

    plotter = pv.Plotter(title="Femur Comparison (Gradient)", window_size=(1200,800))
    plotter.add_mesh(mesh, scalars="ComparisonColor", rgb=True, smooth_shading=True)
    plotter.add_axes()
    plotter.show()
