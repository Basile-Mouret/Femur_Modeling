#!/usr/bin/env python3
"""Test script for Viewer3D with a sample femur mesh."""

import os
import sys

from lib.viewer3D import Viewer3D

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visuFemur.py <path_to_femur_mesh>")
        sys.exit(1)
    viewer = Viewer3D(sys.argv[1])
    viewer.run()
