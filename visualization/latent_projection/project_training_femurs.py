#!/usr/bin/env python3
"""
Project Training Femurs to Latent Space

This script loads all training femurs and projects them into
the latent space of the trained neural network model.
Results are printed and can be saved/visualized.
"""

import numpy as np
import os
import sys
from pathlib import Path

# Add parent visualization directory to path for femur_rdn module
script_dir = Path(__file__).parent.absolute()
visualization_dir = script_dir.parent
sys.path.insert(0, str(visualization_dir))

try:
    import femur_rdn
except ImportError as e:
    print("Error: Could not import femur_rdn module")
    print("Make sure to compile it first:")
    print("  cd build && cmake .. && make femur_rdn")
    print(f"Details: {e}")
    sys.exit(1)


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


def get_training_femurs(training_dir: Path) -> list:
    """
    Get list of all training femur OBJ files.
    """
    femurs = []
    for f in sorted(training_dir.glob("*.obj")):
        femurs.append(f)
    return femurs


def project_femur_to_latent(femur_path: Path) -> np.ndarray:
    """
    Project a single femur to latent space.
    
    Args:
        femur_path: Path to the OBJ file
        
    Returns:
        Latent vector as numpy array
    """
    vertices = load_vertices_from_obj(str(femur_path))
    latent = np.array(femur_rdn.encode(vertices), dtype=np.float64)
    return latent


def main():
    # Paths
    project_root = visualization_dir.parent
    model_path = project_root / "models" / "NeuralNetwork.bin"
    training_dir = project_root / "data" / "training"
    
    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please ensure the trained model exists at this location.")
        sys.exit(1)
    
    # Check if training directory exists
    if not training_dir.exists():
        print(f"Error: Training directory not found: {training_dir}")
        sys.exit(1)
    
    # Initialize the neural network
    print(f"Loading model from: {model_path}")
    femur_rdn.init_decoder(str(model_path))
    
    latent_size = femur_rdn.get_latent_size()
    num_points = femur_rdn.get_num_points()
    
    print(f"Latent space size: {latent_size}")
    print(f"Number of vertices per femur: {num_points}")
    print()
    
    # Get all training femurs
    femur_files = get_training_femurs(training_dir)
    print(f"Found {len(femur_files)} training femurs")
    print("=" * 80)
    print()
    
    # Project each femur and collect results
    results = {}
    all_latents = []
    
    for femur_path in femur_files:
        femur_name = femur_path.stem  # Filename without extension
        
        try:
            latent = project_femur_to_latent(femur_path)
            results[femur_name] = latent
            all_latents.append(latent)
            
            # Print latent vector
            print(f"📌 {femur_name}")
            print(f"   Latent vector: [{', '.join([f'{v:+.4f}' for v in latent])}]")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {femur_name}: {e}")
            print()
    
    # Convert to numpy array for statistics
    all_latents = np.array(all_latents)
    
    # Print summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    print(f"Total femurs processed: {len(results)}")
    print(f"Latent space dimension: {latent_size}")
    print()
    
    print("Per-dimension statistics:")
    print("-" * 60)
    print(f"{'Dim':<6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    
    for i in range(latent_size):
        dim_values = all_latents[:, i]
        print(f"z{i:<5} {np.mean(dim_values):>+10.4f} {np.std(dim_values):>10.4f} "
              f"{np.min(dim_values):>+10.4f} {np.max(dim_values):>+10.4f}")
    
    print("-" * 60)
    print()
    
    # Save results to file
    output_file = script_dir / "latent_projections.npz"
    np.savez(output_file, 
             latents=all_latents,
             femur_names=np.array(list(results.keys())))
    print(f"✅ Results saved to: {output_file}")
    
    # Also save as readable text file
    txt_output = script_dir / "latent_projections.txt"
    with open(txt_output, 'w') as f:
        f.write("Training Femurs Latent Space Projections\n")
        f.write("=" * 80 + "\n\n")
        
        for femur_name, latent in results.items():
            f.write(f"{femur_name}\n")
            f.write(f"  [{', '.join([f'{v:+.6f}' for v in latent])}]\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        for i in range(latent_size):
            dim_values = all_latents[:, i]
            f.write(f"z{i}: mean={np.mean(dim_values):+.6f}, "
                   f"std={np.std(dim_values):.6f}, "
                   f"range=[{np.min(dim_values):+.6f}, {np.max(dim_values):+.6f}]\n")
    
    print(f"✅ Text report saved to: {txt_output}")
    
    return results, all_latents


if __name__ == "__main__":
    results, latents = main()
