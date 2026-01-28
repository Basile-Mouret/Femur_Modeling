# LDDMM Implementation Plan

Migration from emlddmm (displacement approximation) to scikit-shapes (true LDDMM).

## Overview

| Current State | Target State |
|---------------|--------------|
| Displacement: `momentum = target - source` | True LDDMM geodesic shooting |
| emlddmm (image-centric, hacky adaptation) | scikit-shapes (shape-native) |
| Misleading "LDDMM" label | Mathematically correct implementation |

## Phase 1: Registration Module

### File: `registration.py`

**Delete:** Entire `LDDMMPointRegistration` class and `get_device_info()`.

**Create:** New `LDDMMRegistration` class:

```python
@dataclass
class RegistrationResult:
    momentum: np.ndarray        # Initial momentum (N, 3)
    transformed: np.ndarray     # Deformed source (N, 3)
    path: List[np.ndarray]      # Geodesic path
    energy: float               # Deformation energy

class LDDMMRegistration:
    def __init__(self, config: LDDMMConfig = None): ...
    def register(self, source: np.ndarray, target: np.ndarray) -> RegistrationResult: ...
    def compute_momentum(self, source: np.ndarray, target: np.ndarray) -> np.ndarray: ...
```

**Key implementation details:**
- Use `sks.PolyData(points=vertices)` for shape representation
- Use `sks.L2Loss()` since we have point correspondence
- Use `sks.ExtrinsicDeformation(n_steps=n_steps, kernel=kernel, scale=scale)`
- Extract initial momentum from registration parameter

---

## Phase 2: Atlas Builder

### File: `atlas.py`

**Refactor:** `LDDMMAtlasBuilder` → `AtlasBuilder`

**Key changes:**
1. Add `method: Literal['arithmetic', 'geodesic'] = 'arithmetic'` parameter
2. Add comprehensive docstring explaining Euclidean equivalence
3. Remove `registration_params` dict in favor of `LDDMMConfig`
4. Simplify return: `AtlasResult` dataclass instead of dict

```python
@dataclass
class AtlasResult:
    atlas: np.ndarray           # (N, 3) mean shape
    momenta: np.ndarray         # (K, N, 3) initial momenta
    convergence_history: List[float]

class AtlasBuilder:
    def __init__(
        self,
        config: LDDMMConfig = None,
        method: Literal['arithmetic', 'geodesic'] = 'arithmetic',
        max_iterations: int = 10,
        convergence_tol: float = 1e-4,
    ): ...
    
    def build(self, shapes: List[np.ndarray]) -> AtlasResult: ...
    def save(self, path: str) -> None: ...
    
    @classmethod
    def load(cls, path: str) -> 'AtlasBuilder': ...
```

---

## Phase 3: Tangent PCA

### File: `tangent_pca.py`

**Simplifications:**
- Remove `project_momenta()` and `project_shape()` → single `project()`
- Remove `inverse_transform_momenta()` → handled by `synthesize_shape()`
- Use registration module for log map instead of displacement

```python
class TangentPCA:
    def __init__(self, n_components: int = None): ...
    
    def fit(self, atlas: np.ndarray, momenta: np.ndarray) -> 'TangentPCA': ...
    def project(self, shape_or_momentum: np.ndarray) -> np.ndarray: ...
    def synthesize_shape(self, coefficients: np.ndarray) -> np.ndarray: ...
    def synthesize_along_mode(self, mode: int, t_values: List[float]) -> List[np.ndarray]: ...
    def get_mode_extremes(self, mode: int, n_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray]: ...
    
    def save(self, path: str) -> None: ...
    
    @classmethod
    def load(cls, path: str) -> 'TangentPCA': ...
```

---

## Phase 4: Configuration Module

### File: `config.py` (NEW)

```python
@dataclass
class LDDMMConfig:
    n_steps: int = 5              # Geodesic integration steps
    kernel: str = "gaussian"      # RKHS kernel type
    scale: float = 10.0           # Kernel bandwidth σ (mm)
    regularization_weight: float = 0.01
    n_iter: int = 100             # Optimizer iterations
    device: str = "auto"          # "auto", "cuda", "cpu"
```

---

## Phase 5: Tests

Add tests for:
1. Geodesic consistency (round-trip)
2. Momentum shape verification
3. True LDDMM vs displacement comparison

---

## Phase 6: Documentation

Full `README.md` rewrite with theory primer and API reference.

---

## Migration Checklist

- [ ] Install scikit-shapes: `pip install skshapes`
- [ ] Create `config.py`
- [ ] Rewrite `registration.py`
- [ ] Refactor `atlas.py`
- [ ] Update `tangent_pca.py`
- [ ] Update all tests
- [ ] Update visualization scripts
- [ ] Update `README.md`
- [ ] Update `__init__.py`

---

## Breaking Changes Summary

| Old | New |
|-----|-----|
| `LDDMMPointRegistration` | `LDDMMRegistration` |
| `mode='displacement'` | Removed (true LDDMM only) |
| `registration_params` dict | `LDDMMConfig` object |
| `LDDMMAtlasBuilder` | `AtlasBuilder` |
| `.build()` returns `dict` | Returns `AtlasResult` |
| `project_momenta()` / `project_shape()` | `project()` |
