import numpy as np
import os

# Path to the file listing all OBJ files
data_list_path = os.path.join(os.path.dirname(__file__), 'femur_obj_files.txt')

# Read all OBJ file paths
def get_obj_file_list(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Parse vertices from an OBJ file (ignoring normals and faces)
def load_vertices(obj_path):
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) == 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

# Main logic
def main():
    obj_files = get_obj_file_list(data_list_path)
    all_vertices = []
    for i, obj_path in enumerate(obj_files):
        verts = load_vertices(obj_path)
        if i == 0:
            n_points = verts.shape[0]
            mean_vertices = np.zeros_like(verts)
        else:
            assert verts.shape[0] == n_points, f"File {obj_path} has a different number of vertices!"
        mean_vertices += verts
    mean_vertices /= len(obj_files)
    # Save mean femur as OBJ (vertices only)
    out_path = os.path.join(os.path.dirname(__file__), '../data/mean_femur.obj')
    with open(out_path, 'w') as f:
        f.write(f"# {n_points} vertice(s)\n")
        for v in mean_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    print(f"Mean femur saved to {out_path}")
    # Compute the maximum absolute value difference from the mean femur
    max_diff = 0.0
    for obj_path in obj_files:
        verts = load_vertices(obj_path)
        diff = np.abs(verts - mean_vertices)
        max_diff = max(max_diff, np.max(diff))
    print(f"Maximum absolute value difference with mean femur: {max_diff}")

if __name__ == '__main__':
    main()
