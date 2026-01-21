#!/usr/bin/env python3
"""Tests for LDDMM registration module."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from lddmm.registration import (
    LDDMMPointRegistration,
    get_device_info
)
from lddmm.data_loader import FemurDataLoader


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def data_dir():
    """Return path to training data directory."""
    return Path(__file__).parent.parent.parent / "data" / "training"


@pytest.fixture
def shapes(data_dir):
    """Load all shapes."""
    loader = FemurDataLoader(str(data_dir))
    shapes, _ = loader.load_all()
    return shapes


@pytest.fixture
def source_target(shapes):
    """Return a pair of shapes for registration."""
    return shapes[0], shapes[1]


# ============================================================================
# Tests
# ============================================================================

class TestGetDeviceInfo:
    """Tests for get_device_info function."""
    
    def test_returns_dict(self):
        """Test that get_device_info returns a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)
    
    def test_has_required_keys(self):
        """Test that device info has required keys."""
        info = get_device_info()
        assert 'cuda_available' in info
        assert 'device_count' in info


class TestLDDMMPointRegistrationDisplacement:
    """Tests for LDDMMPointRegistration in displacement mode."""
    
    def test_init_default_mode(self):
        """Test default initialization uses displacement mode."""
        reg = LDDMMPointRegistration()
        assert reg.mode == 'displacement'
    
    def test_register_returns_dict(self, source_target):
        """Test that register returns a dictionary."""
        source, target = source_target
        reg = LDDMMPointRegistration(mode='displacement', verbose=False)
        result = reg.register(source, target)
        
        assert isinstance(result, dict)
    
    def test_register_result_keys(self, source_target):
        """Test that registration result has expected keys."""
        source, target = source_target
        reg = LDDMMPointRegistration(mode='displacement', verbose=False)
        result = reg.register(source, target)
        
        assert 'momentum' in result
        assert 'transformed' in result
        assert 'displacement' in result
        assert 'success' in result
        assert 'mode' in result
    
    def test_momentum_shape(self, source_target):
        """Test that momentum has correct shape."""
        source, target = source_target
        reg = LDDMMPointRegistration(mode='displacement', verbose=False)
        result = reg.register(source, target)
        
        assert result['momentum'].shape == source.shape
    
    def test_displacement_equals_momentum(self, source_target):
        """Test that displacement equals momentum in displacement mode."""
        source, target = source_target
        reg = LDDMMPointRegistration(mode='displacement', verbose=False)
        result = reg.register(source, target)
        
        np.testing.assert_array_equal(result['displacement'], result['momentum'])
    
    def test_transformed_equals_target(self, source_target):
        """Test that transformed equals target in displacement mode."""
        source, target = source_target
        reg = LDDMMPointRegistration(mode='displacement', verbose=False)
        result = reg.register(source, target)
        
        np.testing.assert_array_almost_equal(result['transformed'], target)
    
    def test_success_flag(self, source_target):
        """Test that registration succeeds."""
        source, target = source_target
        reg = LDDMMPointRegistration(mode='displacement', verbose=False)
        result = reg.register(source, target)
        
        assert result['success'] is True


class TestLDDMMPointRegistrationEmlddmm:
    """Tests for LDDMMPointRegistration in emlddmm mode."""
    
    def test_init_emlddmm_mode(self):
        """Test initialization in emlddmm mode."""
        reg = LDDMMPointRegistration(mode='emlddmm')
        assert reg.mode == 'emlddmm'
    
    def test_register_returns_result(self, source_target):
        """Test that emlddmm mode returns a result (may fallback)."""
        source, target = source_target
        reg = LDDMMPointRegistration(
            mode='emlddmm',
            n_iter=10,  # Few iterations for speed
            verbose=False
        )
        result = reg.register(source, target)
        
        # Should succeed (either with emlddmm or fallback)
        assert result['success'] is True
        assert result['momentum'].shape == source.shape


class TestComputeMomentum:
    """Tests for compute_momentum method."""
    
    def test_momentum_is_displacement(self, source_target):
        """Test that momentum equals displacement."""
        source, target = source_target
        reg = LDDMMPointRegistration(verbose=False)
        
        momentum = reg.compute_momentum(source, target)
        expected = target - source
        
        np.testing.assert_array_equal(momentum, expected)


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
