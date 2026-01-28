"""
LDDMM-based Shape Analysis Module.

This module provides tools for statistical shape analysis using
Large Deformation Diffeomorphic Metric Mapping (LDDMM):

- **Data loading**: Load OBJ meshes with point correspondence
- **Registration**: LDDMM geodesic shooting via scikit-shapes
- **Atlas building**: Compute population mean (Fréchet mean)
- **Tangent PCA**: Principal component analysis on shape manifold

Theory:
    See LDDMM_THEORY.md for mathematical background on geodesic
    shooting, RKHS kernels, and tangent space statistics.

Dependencies:
    - skshapes (scikit-shapes for LDDMM)
    - torch (PyTorch for GPU computation)
    - numpy (numerical operations)
    - trimesh (OBJ loading)

Example:
    >>> from lddmm import FemurDataLoader, AtlasBuilder, TangentPCA, LDDMMConfig
    >>>
    >>> # Load data
    >>> loader = FemurDataLoader("data/training")
    >>> shapes, filenames = loader.load_all()
    >>>
    >>> # Build atlas
    >>> builder = AtlasBuilder(method='arithmetic')
    >>> result = builder.build(shapes)
    >>>
    >>> # Compute Tangent PCA
    >>> pca = TangentPCA(n_components=10)
    >>> pca.fit(result.atlas, result.momenta)
    >>>
    >>> # Synthesize new shapes
    >>> new_shape = pca.synthesize_along_mode(mode=0, t_values=[-2, 0, 2])
"""

from .config import LDDMMConfig
from .data_loader import FemurDataLoader, verify_correspondence, compute_bounding_box
from .registration import LDDMMRegistration, RegistrationResult
from .atlas import AtlasBuilder, AtlasResult
from .tangent_pca import TangentPCA

__all__ = [
    # Configuration
    "LDDMMConfig",
    # Data loading
    "FemurDataLoader",
    "verify_correspondence",
    "compute_bounding_box",
    # Registration
    "LDDMMRegistration",
    "RegistrationResult",
    # Atlas
    "AtlasBuilder",
    "AtlasResult",
    # PCA
    "TangentPCA",
]

__version__ = "0.2.0"
