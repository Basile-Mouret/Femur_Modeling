#!/usr/bin/env python3
"""Test script for Viewer3D with a sample femur mesh."""

import os
import sys

# Adjust the import path to include the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from viewer3D import Viewer3D

DATA_DIR = "../../build"
FEMUR_FILE = "reconstructed_femur2.obj"

if __name__ == "__main__":
    viewer = Viewer3D(os.path.join(DATA_DIR, FEMUR_FILE))
    viewer.run()
