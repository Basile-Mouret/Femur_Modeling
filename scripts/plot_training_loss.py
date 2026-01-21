import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file

# Scaling factor from main.cpp
scaling_factor = 36.0

with open('epochs_clean.csv', 'r') as f:
    line = f.readline().strip()
    losses = np.array([float(x) for x in line.split(',') if x])

# Convert MSE (on scaled data) to RMSE in mm
rmse_mm = np.sqrt(losses * (scaling_factor ** 2))

# X-axis: epoch numbers (every 10 epochs)
epochs = np.arange(0, len(losses) * 10, 10)


# Losses measured every 50 epochs (MSE on scaled data)
epoch_plus_5000 = np.sqrt(0.000038*scaling_factor**2)
epoch_plus_8000 = np.sqrt(0.000011*scaling_factor**2)

epoch_50 = [0.000435,0.000426,0.000417,0.000409,0.000400,0.000392,0.000384,0.000376,0.000368,0.000361,0.000353,0.000353,0.000346,0.000339,0.000332,0.000325,0.000318,0.000311,0.000304,0.000298,0.000292,0.000286]



# Convert to RMSE in mm
rmse_mm_50 = np.sqrt(np.array(epoch_50) * (scaling_factor ** 2))

# Concatenate epochs and losses for a continuous plot
epochs_10 = np.arange(0, len(losses) * 10, 10)
epochs_50 = np.arange(epochs_10[-1] + 50, epochs_10[-1] + 50 * (len(epoch_50) + 1), 50)


all_epochs = np.concatenate([epochs_10, epochs_50])
all_rmse = np.concatenate([rmse_mm, rmse_mm_50])

# Now that all_epochs is defined, add final loss points after the last epoch
last_epoch = all_epochs[-1]
final_epochs = np.array([last_epoch + 5000, last_epoch + 8000])
final_losses = np.array([0.000038, 0.000011])
final_rmse = np.sqrt(final_losses * (scaling_factor ** 2))




plt.figure(figsize=(10, 6))
plt.plot(all_epochs, all_rmse, marker='x', linestyle='-', color='b')

# Plot the final loss points
plt.scatter(final_epochs, final_rmse, color='r', label='Final Losses', zorder=5)

plt.xlabel('Epoch')
plt.ylabel('RMSE (mm)')
plt.yscale('log')
plt.title('Neural Network Training RMSE')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

