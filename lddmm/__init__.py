"""
LDDMM-based Tangent PCA Module for Femur Shape Analysis.

This module provides tools for:
- Loading femur OBJ files with point correspondence
- LDDMM registration using emlddmm
- Atlas building (Fréchet mean computation)
- Tangent PCA on initial momenta
- Interactive visualization of shape modes

Dependencies:
- torch (PyTorch for GPU computation)
- numpy, scipy (numerical operations)
- trimesh (OBJ loading)
- pyvista (3D visualization)
- emlddmm (LDDMM implementation from lib/emlddmm)

Usage:
    from lddmm import FemurDataLoader, TangentPCA, LDDMMAtlasBuilder
    
    # Load data
    loader = FemurDataLoader("data/training")
    shapes, filenames = loader.load_all()
    
    # Build atlas
    builder = LDDMMAtlasBuilder()
    builder.build(shapes)
    
    # Compute Tangent PCA
    pca = TangentPCA(n_components=10)
    pca.fit(builder.atlas, builder.momenta)
"""

from .data_loader import FemurDataLoader, verify_correspondence
from .registration import LDDMMPointRegistration
from .atlas import LDDMMAtlasBuilder
from .tangent_pca import TangentPCA

__all__ = [
    'FemurDataLoader',
    'verify_correspondence', 
    'LDDMMPointRegistration',
    'LDDMMAtlasBuilder',
    'TangentPCA',
]

__version__ = '0.1.0'
