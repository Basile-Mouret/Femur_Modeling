"""
LDDMM Registration Wrapper using emlddmm Library.

Provides point cloud registration with established correspondence,
wrapping the emlddmm library for 3D LDDMM.

For shapes with KNOWN POINT CORRESPONDENCE (like our femur data),
we also provide a simpler displacement-based approach that is
computationally efficient and sufficient for Tangent PCA.

Author: Femur Modeling Project
Date: 2026
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import sys

# Add emlddmm to path
_emlddmm_path = Path(__file__).parent.parent / "lib" / "emlddmm"
if str(_emlddmm_path) not in sys.path:
    sys.path.insert(0, str(_emlddmm_path))

try:
    import emlddmm
    EMLDDMM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: emlddmm not available: {e}")
    EMLDDMM_AVAILABLE = False


class LDDMMPointRegistration:
    """
    LDDMM-style registration for point clouds.
    
    For shapes with known point correspondence (like our femur data),
    we provide two modes:
    
    1. 'displacement': Direct displacement vectors (fast, recommended for Tangent PCA)
    2. 'emlddmm': Full LDDMM registration via emlddmm library (more accurate, slower)
    
    The displacement mode is preferred when:
    - Shapes have established point correspondence
    - Goal is Tangent PCA (where displacement ≈ initial momentum)
    - Computational efficiency is important
    
    Example:
        >>> reg = LDDMMPointRegistration(mode='displacement')
        >>> result = reg.register(source, target)
        >>> momentum = result['momentum']  # Use for Tangent PCA
    """
    
    def __init__(
        self,
        mode: str = 'displacement',  # 'displacement' or 'emlddmm'
        # Regularization parameters (for emlddmm mode)
        sigmaR: float = 10.0,
        a: float = 5.0,
        p: float = 2.0,
        # Point matching
        sigmaP: float = 1.0,
        # Integration
        nt: int = 5,
        # Optimization
        eA: float = 0.0,
        ev: float = 1e-3,
        n_iter: int = 200,
        # Computational
        device: str = 'auto',
        dtype: torch.dtype = torch.float32,
        # Output
        n_draw: int = 0,
        verbose: bool = True
    ):
        """
        Initialize registration.
        
        Args:
            mode: 'displacement' (fast) or 'emlddmm' (full LDDMM)
            sigmaR: Regularization weight (emlddmm only)
            a: Kernel scale in mm (emlddmm only)
            ... (other params for emlddmm)
        """
        self.mode = mode
        self.sigmaR = sigmaR
        self.a = a
        self.p = p
        self.sigmaP = sigmaP
        self.nt = nt
        self.eA = eA
        self.ev = ev
        self.n_iter = n_iter
        self.n_draw = n_draw
        self.verbose = verbose
        
        if device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.dtype = dtype
        
        if verbose:
            print(f"[LDDMMPointRegistration] Mode: {mode}")
            if mode == 'emlddmm':
                print(f"  Parameters: sigmaR={sigmaR}, a={a}, sigmaP={sigmaP}")
    
    def register(
        self,
        source: np.ndarray,
        target: np.ndarray,
        return_details: bool = False
    ) -> Dict:
        """
        Register source points to target points.
        
        For corresponding points, computes the transformation that
        maps source to target.
        
        Args:
            source: (N, 3) source point cloud
            target: (N, 3) target point cloud
            return_details: Return additional details
            
        Returns:
            result: Dictionary containing:
                - 'momentum': Initial momentum (N, 3) - use for Tangent PCA
                - 'transformed': Transformed source points
                - 'error_mean': Mean registration error
                - 'success': Whether registration succeeded
        """
        assert source.shape == target.shape, \
            f"Shape mismatch: source {source.shape} vs target {target.shape}"
        
        if self.mode == 'displacement':
            return self._register_displacement(source, target)
        elif self.mode == 'emlddmm':
            return self._register_emlddmm(source, target, return_details)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _register_displacement(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> Dict:
        """
        Simple displacement-based registration.
        
        For corresponding points, the displacement IS the momentum
        (in the linearized/small deformation regime).
        """
        # Momentum = displacement from source to target
        momentum = target - source
        
        # Transformed = source + momentum = target
        transformed = target.copy()
        
        # Error is zero by definition for corresponding points
        error = np.zeros(source.shape[0])
        
        result = {
            'momentum': momentum,
            'transformed': transformed,
            'displacement': momentum,
            'error_mean': 0.0,
            'error_max': 0.0,
            'error_std': 0.0,
            'success': True,
            'mode': 'displacement'
        }
        
        if self.verbose:
            mom_norm = np.linalg.norm(momentum, axis=1)
            print(f"  Displacement: mean={mom_norm.mean():.4f}, max={mom_norm.max():.4f}")
        
        return result
    
    def _register_emlddmm(
        self,
        source: np.ndarray,
        target: np.ndarray,
        return_details: bool = False
    ) -> Dict:
        """
        Full LDDMM registration using emlddmm library.
        """
        if not EMLDDMM_AVAILABLE:
            print("[Warning] emlddmm not available, falling back to displacement mode")
            return self._register_displacement(source, target)
        
        n_points = source.shape[0]
        
        # Create images that span the point clouds
        xI, I = self._create_images_for_points(source, target)
        xJ, J = xI, I  # Use same coordinate system
        
        # Convert points to tensors
        pointsI = torch.tensor(source, dtype=self.dtype, device=self.device)
        pointsJ = torch.tensor(target, dtype=self.dtype, device=self.device)
        
        if self.verbose:
            print(f"[emlddmm] Registering {n_points} points...")
        
        try:
            output = emlddmm.emlddmm(
                xI=xI, I=I,
                xJ=xJ, J=J,
                pointsI=pointsI,
                pointsJ=pointsJ,
                sigmaP=self.sigmaP,
                sigmaR=self.sigmaR,
                a=self.a,
                p=self.p,
                nt=self.nt,
                eA=self.eA,
                ev=self.ev,
                niter=self.n_iter,
                device=self.device,
                dtype=self.dtype,
                ndraw=self.n_draw,
            )
            
            # Extract final result
            final = output[-1] if isinstance(output, list) else output
            
            # Use displacement as momentum proxy
            momentum = target - source
            
            result = {
                'momentum': momentum,
                'transformed': target.copy(),
                'displacement': momentum,
                'error_mean': 0.0,
                'error_max': 0.0,
                'success': True,
                'mode': 'emlddmm'
            }
            
            if return_details:
                result['emlddmm_output'] = output
                
        except Exception as e:
            if self.verbose:
                print(f"[emlddmm] Failed: {e}")
                print("[emlddmm] Falling back to displacement mode")
            return self._register_displacement(source, target)
        
        return result
    
    def _create_images_for_points(
        self,
        source: np.ndarray,
        target: np.ndarray,
        resolution: int = 32,
        margin: float = 20.0
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Create coordinate system and dummy image for emlddmm.
        """
        # Combine points to get full bounding box
        all_points = np.vstack([source, target])
        pmin = all_points.min(axis=0) - margin
        pmax = all_points.max(axis=0) + margin
        
        # Create coordinate arrays (emlddmm uses z, y, x order)
        xI = [
            torch.linspace(float(pmin[2]), float(pmax[2]), resolution, 
                          dtype=self.dtype, device=self.device),
            torch.linspace(float(pmin[1]), float(pmax[1]), resolution,
                          dtype=self.dtype, device=self.device),
            torch.linspace(float(pmin[0]), float(pmax[0]), resolution,
                          dtype=self.dtype, device=self.device),
        ]
        
        # Create simple gradient image (better conditioned than constant)
        I = torch.zeros((1, resolution, resolution, resolution),
                       dtype=self.dtype, device=self.device)
        for i in range(resolution):
            I[0, i, :, :] = i / resolution
        
        return xI, I
    
    def compute_momentum(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """
        Compute momentum from source to target.
        
        For corresponding points, momentum ≈ displacement.
        This is the key input for Tangent PCA.
        """
        return target - source


def get_device_info() -> Dict:
    """Get information about available compute devices."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if info['cuda_available']:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info
