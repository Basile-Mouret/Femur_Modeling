import matplotlib.pyplot as plt
import sys
import os

def plot_epoch_times(filenames, labels=None, title="Epoch Times Comparison"):
    plt.figure(figsize=(8, 5))
    for i, fname in enumerate(filenames):
        if not os.path.exists(fname):
            print(f"File not found: {fname}")
            continue
        with open(fname, 'r') as f:
            times = [float(line.strip()) for line in f if line.strip()]
        label = labels[i] if labels and i < len(labels) else os.path.basename(fname)
        plt.plot(range(1, len(times)+1), times, marker='o', label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot epoch times from one or more .txt files.")
    parser.add_argument('files', nargs='+', help='List of epoch_times.txt files to plot')
    parser.add_argument('--labels', nargs='*', help='Labels for each curve (optional)')
    parser.add_argument('--title', default='Epoch Times Comparison', help='Plot title')
    args = parser.parse_args()
    plot_epoch_times(args.files, labels=args.labels, title=args.title)
