import pandas as pd
import matplotlib.pyplot as plt

ith_datapoint = 100

# Load the standardized predictions and real values
filepath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{ith_datapoint}_DP_predictions_vs_real.csv'
data = pd.read_csv(filepath)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
features = ['X', 'Y', 'Yaw']
for i, feature in enumerate(features):
    axs[i].plot(data[f'Predicted_{feature}'], label=f'Predicted {feature}')
    axs[i].plot(data[f'Real_{feature}'], label=f'Real {feature}', alpha=0.7)
    axs[i].set_title(f'Predicted vs Real {feature}')
    axs[i].legend()

plt.tight_layout()
plt.show()
