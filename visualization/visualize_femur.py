#!/usr/bin/env python3
"""Femur Mesh Visualizer"""

import os
import sys

# Adjust the import path to include the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from viewer3D import Viewer3D


if __name__ == "__main__":
    viewer = Viewer3D(sys.argv[1])
    viewer.run()
