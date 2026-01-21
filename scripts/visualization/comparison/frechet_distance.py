#!/usr/bin/env python3
"""
Fréchet (Geodesic) Distance Between Femur Shapes

Computes the geodesic distance between two femur shapes in the LDDMM 
diffeomorphism space. This measures the "deformation energy" required
to transform one shape into another.

Usage:
    python frechet_distance.py <femur1.obj> <femur2.obj> [--sigma 10.0]
    
The geodesic distance is computed as:
    d(S1, S2) = ||v||_V = sqrt(<v, Kv>)
    
where v is the optimal initial momentum field and K is the RKHS kernel.

Author: Generated for Femur Modeling Project
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import numpy as np
import torch
import trimesh
from typing import Tuple, Dict, Optional
import pyvista as pv


def load_obj_vertices(obj_path: str) -> np.ndarray:
    """Load vertices from OBJ file."""
    mesh = trimesh.load(obj_path, process=False)
    return np.array(mesh.vertices, dtype=np.float32)


def load_obj_mesh(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from OBJ file."""
    mesh = trimesh.load(obj_path, process=False)
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute Gaussian kernel matrix K(x, y).
    
    K(x_i, y_j) = exp(-||x_i - y_j||^2 / (2 * sigma^2))
    
    Uses chunked computation for memory efficiency with large point clouds.
    """
    N = x.shape[0]
    M = y.shape[0]
    
    # For small matrices, compute directly
    if N * M < 1e7:  # ~10M elements threshold
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, 3)
        sq_dist = (diff ** 2).sum(dim=2)  # (N, M)
        return torch.exp(-sq_dist / (2 * sigma ** 2))
    
    # For large matrices, compute in chunks
    chunk_size = max(1, int(1e7 / M))
    K = torch.zeros(N, M, device=x.device, dtype=x.dtype)
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        diff = x[i:end_i].unsqueeze(1) - y.unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=2)
        K[i:end_i] = torch.exp(-sq_dist / (2 * sigma ** 2))
    
    return K


def compute_rkhs_norm(momentum: torch.Tensor, points: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute RKHS norm of momentum field: ||v||_V = sqrt(<p, Kp>)
    
    Args:
        momentum: (N, 3) momentum vectors at each point
        points: (N, 3) point positions
        sigma: Kernel bandwidth
        
    Returns:
        RKHS norm (scalar)
    """
    K = gaussian_kernel(points, points, sigma)  # (N, N)
    
    # <p, Kp> = sum over dimensions of p^T K p
    # For 3D: sum of p_x^T K p_x + p_y^T K p_y + p_z^T K p_z
    norm_sq = 0.0
    for d in range(3):
        p_d = momentum[:, d]  # (N,)
        norm_sq = norm_sq + p_d @ K @ p_d
    
    return torch.sqrt(norm_sq)


def compute_varifold_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """
    Compute varifold distance between point clouds.
    This is used as the data attachment term.
    """
    K_ss = gaussian_kernel(source, source, sigma).sum()
    K_tt = gaussian_kernel(target, target, sigma).sum()
    K_st = gaussian_kernel(source, target, sigma).sum()
    
    return K_ss + K_tt - 2 * K_st


def shooting(
    points: torch.Tensor,
    momentum: torch.Tensor,
    sigma: float,
    n_steps: int = 10
) -> torch.Tensor:
    """
    Geodesic shooting: integrate the Hamiltonian equations.
    
    Returns the deformed points at t=1.
    """
    dt = 1.0 / n_steps
    q = points.clone()
    p = momentum.clone()
    
    for _ in range(n_steps):
        K = gaussian_kernel(q, q, sigma)
        
        # Velocity: v = Kp
        v = torch.zeros_like(q)
        for d in range(3):
            v[:, d] = K @ p[:, d]
        
        # Update position
        q = q + dt * v
        
        # Update momentum (simplified Euler, full Hamiltonian would need gradient of K)
        # For geodesic distance computation, we use the initial momentum norm
        
    return q


def optimize_momentum(
    source: np.ndarray,
    target: np.ndarray,
    sigma: float = 10.0,
    lambda_reg: float = 1.0,
    n_iterations: int = 100,
    lr: float = 0.1,
    n_shooting_steps: int = 10,
    verbose: bool = True,
    use_cpu: bool = True  # Force CPU for large point clouds
) -> Tuple[np.ndarray, float, Dict]:
    """
    Find optimal momentum field to deform source to target.
    
    The geodesic distance is the RKHS norm of this optimal momentum.
    
    Args:
        source: (N, 3) source point cloud
        target: (N, 3) target point cloud
        sigma: Kernel bandwidth
        lambda_reg: Regularization weight
        n_iterations: Optimization iterations
        lr: Learning rate
        n_shooting_steps: Integration steps for shooting
        verbose: Print progress
        use_cpu: Force CPU computation (recommended for large meshes)
        
    Returns:
        Tuple of (optimal_momentum, geodesic_distance, info_dict)
    """
    # For large point clouds, force CPU to avoid OOM
    if use_cpu or len(source) > 5000:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    source_t = torch.tensor(source, dtype=torch.float32, device=device)
    target_t = torch.tensor(target, dtype=torch.float64, device=device)  # Use float64 for stability
    source_t = source_t.double()
    
    # Compute kernel matrix once (add small diagonal for numerical stability)
    K = gaussian_kernel(source_t, source_t, sigma)
    K = K + 1e-6 * torch.eye(K.shape[0], device=device, dtype=torch.float64)
    
    # Solve directly: we want momentum p such that K @ p ≈ (target - source)
    # This is the linearized matching problem
    displacement = target_t - source_t  # (N, 3)
    
    # Solve K @ p = displacement for each dimension
    # Using least squares: p = (K^T K + lambda I)^{-1} K^T d
    KtK = K.T @ K
    reg_matrix = KtK + lambda_reg * torch.eye(K.shape[0], device=device, dtype=torch.float64)
    
    momentum = torch.zeros_like(source_t)
    for d in range(3):
        rhs = K.T @ displacement[:, d]
        momentum[:, d] = torch.linalg.solve(reg_matrix, rhs)
    
    # Compute geodesic distance = ||p||_V = sqrt(p^T K p)
    with torch.no_grad():
        final_geodesic = compute_rkhs_norm(momentum, source_t, sigma).item()
        
        # Compute deformed for visualization
        velocity = torch.zeros_like(source_t)
        for d in range(3):
            velocity[:, d] = K @ momentum[:, d]
        final_deformed = source_t + velocity
        
        # Compute reconstruction error
        recon_error = ((final_deformed - target_t) ** 2).sum().sqrt().item()
    
    if verbose:
        print(f"  Direct solve completed")
        print(f"  Reconstruction error: {recon_error:.4f}")
        print(f"  Geodesic distance: {final_geodesic:.4f}")
    
    history = {"loss": [recon_error], "data_loss": [recon_error], "reg_loss": [0], "geodesic_dist": [final_geodesic]}
    
    return (
        momentum.detach().cpu().numpy(),
        final_geodesic,
        {
            "history": history,
            "deformed": final_deformed.cpu().numpy(),
            "recon_error": recon_error,
        }
    )


def compute_frechet_distance(
    femur1_path: str,
    femur2_path: str,
    sigma: float = 10.0,
    n_iterations: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Compute Fréchet (geodesic) distance between two femur shapes.
    
    Args:
        femur1_path: Path to first femur OBJ
        femur2_path: Path to second femur OBJ
        sigma: Kernel bandwidth (controls deformation smoothness)
        n_iterations: Optimization iterations
        verbose: Print progress
        subsample: Subsample factor (e.g., 10 = use every 10th point)
        
    Returns:
        Dictionary with distance metrics and deformation info
    """
    print(f"\nComputing Fréchet distance between:")
    print(f"  Shape 1: {Path(femur1_path).name}")
    print(f"  Shape 2: {Path(femur2_path).name}")
    print(f"  Kernel sigma: {sigma}")
    print()
    
    # Load shapes
    vertices1_full = load_obj_vertices(femur1_path)
    vertices2_full = load_obj_vertices(femur2_path)
    
    if vertices1_full.shape != vertices2_full.shape:
        raise ValueError(f"Shape mismatch: {vertices1_full.shape} vs {vertices2_full.shape}. "
                        "Shapes must have point correspondence.")
    
    n_points_full = len(vertices1_full)
    print(f"  Points per shape: {n_points_full}")
    
    # Also compute L2 distance on full resolution
    l2_per_vertex = np.linalg.norm(vertices1_full - vertices2_full, axis=1)
    l2_total = np.sqrt(np.sum(l2_per_vertex ** 2))
    l2_rmse = np.sqrt(np.mean(l2_per_vertex ** 2))
    
    print(f"\nEuclidean metrics (full resolution):")
    print(f"  L2 total: {l2_total:.4f}")
    print(f"  L2 RMSE:  {l2_rmse:.4f}")
    print(f"  L2 max:   {l2_per_vertex.max():.4f}")
    
    # Subsample for geodesic computation if needed
    subsample = max(1, n_points_full // 2000)  # Target ~2000 points
    if subsample > 1:
        print(f"\n  Subsampling by factor {subsample} for geodesic computation...")
        vertices1 = vertices1_full[::subsample]
        vertices2 = vertices2_full[::subsample]
        print(f"  Using {len(vertices1)} points")
    else:
        vertices1 = vertices1_full
        vertices2 = vertices2_full
    
    # Compute geodesic distance
    print(f"\nOptimizing geodesic path...")
    momentum, geodesic_dist, info = optimize_momentum(
        vertices1, vertices2,
        sigma=sigma,
        n_iterations=n_iterations,
        verbose=verbose
    )
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"  Fréchet (geodesic) distance: {geodesic_dist:.4f}")
    print(f"  Euclidean L2 distance:       {l2_total:.4f}")
    print(f"  Ratio (Fréchet/L2):          {geodesic_dist/l2_total:.4f}")
    print(f"{'='*50}")
    
    return {
        "frechet_distance": geodesic_dist,
        "l2_distance": l2_total,
        "l2_rmse": l2_rmse,
        "l2_max": l2_per_vertex.max(),
        "momentum": momentum,
        "deformed": info["deformed"],
        "vertices1": vertices1_full,
        "vertices2": vertices2_full,
        "l2_per_vertex": l2_per_vertex,
        "history": info["history"],
    }


def visualize_comparison(
    result: Dict,
    femur1_path: str,
    femur2_path: str,
    template_path: Optional[str] = None
):
    """Visualize the two shapes and the deformation."""
    
    # Get faces from one of the meshes
    if template_path:
        _, faces = load_obj_mesh(template_path)
    else:
        _, faces = load_obj_mesh(femur1_path)
    
    vertices1 = result["vertices1"]
    vertices2 = result["vertices2"]
    l2_errors = result["l2_per_vertex"]
    
    def make_mesh(verts, scalars, name):
        pv_faces = np.hstack([[3] + list(f) for f in faces])
        mesh = pv.PolyData(verts, pv_faces)
        mesh[name] = scalars
        return mesh
    
    plotter = pv.Plotter(shape=(1, 2), window_size=(1400, 600))
    
    # Shape 1
    plotter.subplot(0, 0)
    plotter.add_text(f"Shape 1\n{Path(femur1_path).name}", font_size=10)
    mesh1 = make_mesh(vertices1, np.zeros(len(vertices1)), "dummy")
    plotter.add_mesh(mesh1, color="lightblue", smooth_shading=True)
    
    # Shape 2 with error heatmap
    plotter.subplot(0, 1)
    plotter.add_text(f"Shape 2 (error from Shape 1)\nFréchet: {result['frechet_distance']:.3f}, L2: {result['l2_distance']:.3f}", font_size=10)
    mesh2 = make_mesh(vertices2, l2_errors, "L2 Error")
    plotter.add_mesh(mesh2, scalars="L2 Error", cmap="coolwarm", smooth_shading=True,
                     scalar_bar_args={"title": "L2 Error (mm)"})
    
    plotter.link_views()
    plotter.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute Fréchet (geodesic) distance between two femur shapes"
    )
    parser.add_argument("femur1", help="Path to first femur OBJ file")
    parser.add_argument("femur2", help="Path to second femur OBJ file")
    parser.add_argument("--sigma", type=float, default=10.0,
                        help="Kernel bandwidth (controls deformation smoothness)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Optimization iterations")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Resolve paths
    femur1 = args.femur1 if Path(args.femur1).is_absolute() else str(project_root / args.femur1)
    femur2 = args.femur2 if Path(args.femur2).is_absolute() else str(project_root / args.femur2)
    
    # Compute distance
    result = compute_frechet_distance(
        femur1, femur2,
        sigma=args.sigma,
        n_iterations=args.iterations,
        verbose=not args.quiet
    )
    
    # Visualize
    if not args.no_visualize:
        visualize_comparison(result, femur1, femur2)
    
    return result


if __name__ == "__main__":
    main()
