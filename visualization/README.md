# 3D Femur Visualization

Contains tools to visualize in 3D femur meshes.

## Prerequisites

Implemented and tested with **Python 3.12**.

Check your current version:
```bash
python3 --version
```

If you don't have the *venv* module installed:

```bash
sudo apt update
sudo apt install python3-venv
```

## Installation

We use a **Virtual Environment** (venv) :

### 1. Create the virtual environment

```bash
python3 -m venv venv
```

### 2. Activate the environment

You must redo this step every time you open a new terminal to work on the project.

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the visualization

```bash
chmod u+x visualizer.py #if it's the first time
./visualizer.py
```