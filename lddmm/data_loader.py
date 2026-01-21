"""
Data Loader for Femur OBJ Files.

Handles loading of femur meshes from OBJ files with established
point correspondence (all shapes have same vertex count and ordering).

Author: Femur Modeling Project
Date: 2026
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import trimesh


class FemurDataLoader:
    """
    Load femur shapes from OBJ files.
    
    Designed for data with established point correspondence where
    all shapes have the same number of vertices in the same order.
    
    Example:
        >>> loader = FemurDataLoader("data/training")
        >>> shapes, filenames = loader.load_all()
        >>> print(f"Loaded {len(shapes)} shapes with {shapes[0].shape[0]} vertices each")
    """
    
    def __init__(self, data_dir: str, file_pattern: str = "*.obj"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing OBJ files
            file_pattern: Glob pattern for finding files (default: "*.obj")
        """
        self.data_dir = Path(data_dir)
        self.file_pattern = file_pattern
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find all matching files
        self.files = sorted(self.data_dir.glob(file_pattern))
        
        if len(self.files) == 0:
            raise ValueError(f"No files matching '{file_pattern}' found in {self.data_dir}")
        
        print(f"[FemurDataLoader] Found {len(self.files)} files in {self.data_dir}")
    
    def load_single(self, filepath: Path) -> np.ndarray:
        """
        Load a single OBJ file and return vertices.
        
        Args:
            filepath: Path to OBJ file
            
        Returns:
            vertices: (N, 3) array of vertex positions
        """
        # Use trimesh to load OBJ
        mesh = trimesh.load(filepath, process=False)
        
        # Handle both Trimesh and PointCloud types
        if hasattr(mesh, 'vertices'):
            vertices = np.array(mesh.vertices, dtype=np.float32)
        else:
            # It's a point cloud
            vertices = np.array(mesh, dtype=np.float32)
        
        return vertices
    
    def load_all(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load all OBJ files in the data directory.
        
        Returns:
            shapes: List of (N, 3) arrays
            filenames: List of corresponding filenames
        """
        shapes = []
        filenames = []
        
        for filepath in self.files:
            try:
                vertices = self.load_single(filepath)
                shapes.append(vertices)
                filenames.append(filepath.name)
                print(f"  Loaded {filepath.name}: {vertices.shape}")
            except Exception as e:
                print(f"  Warning: Failed to load {filepath.name}: {e}")
        
        return shapes, filenames
    
    def get_file_list(self) -> List[Path]:
        """Return list of found file paths."""
        return self.files
    
    def __len__(self) -> int:
        """Return number of files found."""
        return len(self.files)


def verify_correspondence(shapes: List[np.ndarray], verbose: bool = True) -> bool:
    """
    Verify that all shapes have the same number of vertices (point correspondence).
    
    Args:
        shapes: List of (N, 3) arrays
        verbose: Print verification info
        
    Returns:
        True if all shapes have same vertex count, False otherwise
    """
    if len(shapes) == 0:
        if verbose:
            print("[verify_correspondence] No shapes provided")
        return False
    
    n_vertices = shapes[0].shape[0]
    
    for i, shape in enumerate(shapes):
        if shape.shape[0] != n_vertices:
            if verbose:
                print(f"[verify_correspondence] Shape {i} has {shape.shape[0]} vertices, "
                      f"expected {n_vertices}")
            return False
        if shape.shape[1] != 3:
            if verbose:
                print(f"[verify_correspondence] Shape {i} has {shape.shape[1]} dimensions, "
                      f"expected 3")
            return False
    
    if verbose:
        print(f"[verify_correspondence] ✓ All {len(shapes)} shapes have {n_vertices} vertices")
    
    return True


def compute_bounding_box(shapes: List[np.ndarray]) -> dict:
    """
    Compute bounding box statistics for a collection of shapes.
    
    Args:
        shapes: List of (N, 3) arrays
        
    Returns:
        Dictionary with min, max, center, size
    """
    all_points = np.vstack(shapes)
    
    return {
        'min': all_points.min(axis=0),
        'max': all_points.max(axis=0),
        'center': all_points.mean(axis=0),
        'size': all_points.max(axis=0) - all_points.min(axis=0)
    }
