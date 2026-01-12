#!/usr/bin/env python3
"""
Test script for the 3D Femur Viewer application in point cloud mode.
This script initializes the Viewer3D class with a sample femur .obj file
and runs the interactive visualization in point cloud rendering style.
"""

import os
import sys

# Add the parent directory to the path to import viewer3D
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from viewer3D import Viewer3D

path = "../../data/validation"
femur = "R_Femur_22_DECIM.obj.FINAL.obj"

try:
    app = Viewer3D(os.path.join(path, femur))
    app.run(render_style="points")
except Exception as e:
    print(f"Application Error: {e}")
