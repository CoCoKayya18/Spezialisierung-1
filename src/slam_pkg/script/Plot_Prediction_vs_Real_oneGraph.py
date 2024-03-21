import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ith_datapoint = 40
# isSparse = 'sparse_'
isSparse = ''

# Load the standardized predictions and real values
filepath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isSparse}{ith_datapoint}_DP_predictions_vs_real.csv'
data = pd.read_csv(filepath)

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # Increase subplots for error plots
features = ['X', 'Y', 'Yaw']

for i, feature in enumerate(features):
    # Plot predicted vs real
    axs[i, 0].plot(data[f'Predicted_{feature}'], label=f'Predicted {feature}')
    axs[i, 0].plot(data[f'Real_{feature}'], label=f'Real {feature}', alpha=0.7)
    axs[i, 0].set_title(f'Predicted vs Real {feature}')
    axs[i, 0].legend()

    # Calculate and plot the error
    error = data[f'Predicted_{feature}'] - data[f'Real_{feature}']
    axs[i, 1].plot(error, label=f'Error in {feature}', color='red')
    axs[i, 1].set_title(f'Error in {feature}')
    axs[i, 1].axhline(0, color='blue', linewidth=0.8)  # Add a horizontal line at error = 0
    axs[i, 1].legend()

    # Calculate and plot the mean error for every datapoint
    mean_error = error.expanding().mean()
    axs[i, 2].plot(mean_error, label=f'Mean Error in {feature}', color='green')
    axs[i, 2].set_title(f'Cumulative Mean Error in {feature}')
    axs[i, 2].legend()

plt.tight_layout()
plt.show()