import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    obj_folder = "data/training"
    obj_files = list(Path(obj_folder).glob('*.obj'))

    x = [] 
    y = []
    z = []

    for obj_file in obj_files:
        with open(obj_file, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x.append(float(parts[1]))
                    y.append(float(parts[2]))
                    z.append(float(parts[3]))

    x = np.array(x)    
    y = np.array(y)    
    z = np.array(z)    

    # Print statistics
    print(f"\nStatistics:")
    print(f"  coordinate x: mean = {np.mean(x)}, std = {np.std(x)}")
    print(f"  coordinate y: mean = {np.mean(y)}, std = {np.std(y)}")
    print(f"  coordinate z: mean = {np.mean(z)}, std = {np.std(z)}\n")

    plt.plot(range(len(x)),sorted(x))
    plt.show()


if __name__ == "__main__":
    main()
