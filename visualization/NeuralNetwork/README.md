# Femur Latent Space Visualization

This folder contains scripts to explore and visualize the latent space of the femur neural network autoencoder. Here is a drive with some trained model of autoencoder. (https://drive.google.com/drive/folders/1XfqVVEv_XzGETwcklJSmiH7S0PmQqdPp)

## Main scripts

| Script                   | Purpose                                      |
|--------------------------|----------------------------------------------|
| latent_explorer.py       | Interactive latent space explorer (sliders)  |
| view_latent_3d.py        | 3D plot of all training femurs in latent space|
| project_training_femurs.py | Project all training femurs to latent space |
| compare_femur.py | Compare 2 femurs (per-vertex diff)|
| visuFemur.py             | Quick OBJ mesh viewer                        |
| lib/viewer3D.py          | 3D mesh viewer class (used by other scripts) |

## Requirements

- Python 3.10+
- See requirements.txt for Python packages

## Setup

1. Create and activate the virtual environment (from project root):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2. Install dependencies:
    ```bash
    pip install -r scripts/visualization/requirements.txt
    ```
3. Build the C++ module:
    ```bash
    cd build
    cmake ..
    make femur_rdn
    ```

## Required Files and Data

To run the scripts, you need the following files in the correct locations:

- OBJ meshes in `data/training/` — training femur meshes
- `scripts/visualization/reconstruction_data/base_femur_for_visu.obj` — reference mesh for faces

If your files are elsewhere, update the paths in the scripts accordingly.

## Usage

Run all scripts from scripts/visualization/ with the virtual environment activated.

- Interactive latent explorer:
  ```bash
  ./latent_explorer.py <path_to_neural_network.bin> <path_to_base_femur.obj>  # The base_femur OBJ is encoded to initialize the sliders and is the femur shown at startup (basically the mean_femur)
  ```
- Project all training femurs:
  ```bash
  ./project_training_femurs.py
  ```
- 3D latent space viewer (all femurs):
  ```bash
  ./view_latent_3d.py -n -1
  ```
- Compare femur to mean:
  ```bash
  ./compare_femur.py <mean_femur.obj> <femur.obj>
  ```
- View OBJ mesh:
  ```bash
  ./visuFemur.py <femur.obj>
  ```

**Note:** Always activate the virtual environment and rebuild the C++ module if you change the C++ code.
