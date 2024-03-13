# import pandas as pd
# import matplotlib.pyplot as plt

# # Path to your CSV file
# file_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'

# # Read the CSV file
# data = pd.read_csv(file_path)

# # Plotting
# fig, axs = plt.subplots(3, 2, figsize=(12, 9))

# # X values plot
# axs[0, 0].scatter(data["GroundTruth_X"], data["Delta_X_X"])
# axs[0, 0].set_title('Ground Truth X vs. Delta X')
# axs[0, 0].set_xlabel('Ground Truth X')
# axs[0, 0].set_ylabel('Delta X')

# axs[0, 1].scatter(data["Velocity_Linear_X"], data["Delta_X_X"])
# axs[0, 1].set_title('Velocity_Linear_X vs. Delta X')
# axs[0, 1].set_xlabel('Velocity Linear X')
# axs[0, 1].set_ylabel('Delta X')

# # Y values plot
# axs[1, 0].scatter(data["Ground_Truth_Y"], data["Delta_X_Y"])
# axs[1, 0].set_title('Ground Truth Y vs. Delta Y')
# axs[1, 0].set_xlabel('Ground Truth Y')
# axs[1, 0].set_ylabel('Delta Y')

# axs[1, 1].scatter(data["Velocity_Linear_Y"], data["Delta_X_Y"])
# axs[1, 1].set_title('Velocity Linear Y vs. Delta Y')
# axs[1, 1].set_xlabel('Velocity Linear Y')
# axs[1, 1].set_ylabel('Delta Y')

# # Z values plot
# axs[2, 0].scatter(data["Ground_Truth_Yaw"], data["delta_X_Yaw"])
# axs[2, 0].set_title('Ground Truth Yaw vs. Delta Yaw')
# axs[2, 0].set_xlabel('Ground Truth Yaw')
# axs[2, 0].set_ylabel('Delta Yaw')

# axs[2, 1].scatter(data["Velocity_Angular_Yaw"], data["delta_X_Yaw"])
# axs[2, 1].set_title('Velocity Angular Yaw vs. Delta Yaw')
# axs[2, 1].set_xlabel('Velocity Angular Yaw')
# axs[2, 1].set_ylabel('Delta Yaw')

# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the original data
file_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'
data = pd.read_csv(file_path)

# Standardize the data
features = ['GroundTruth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']
scaler = StandardScaler()

data_standardized = scaler.fit_transform(data[features + target])
data_standardized = pd.DataFrame(data_standardized, columns=features + target)

# Plotting function to plot both original and standardized data
def plot_data(original_data, standardized_data, feature, target, axs, row, col):
    axs[row, col*2].scatter(original_data[feature], original_data[target])
    axs[row, col*2].set_title(f'Original {feature} vs. {target}')
    axs[row, col*2].set_xlabel(feature)
    axs[row, col*2].set_ylabel(target)

    axs[row, col*2+1].scatter(standardized_data[feature], standardized_data[target])
    axs[row, col*2+1].set_title(f'Standardized {feature} vs. {target}')
    axs[row, col*2+1].set_xlabel(feature)
    axs[row, col*2+1].set_ylabel(target)

# Plotting
fig, axs = plt.subplots(3, 4, figsize=(18, 9))  # Adjusted for double the number of columns

plot_data(data, data_standardized, "GroundTruth_X", "Delta_X_X", axs, 0, 0)
plot_data(data, data_standardized, "Velocity_Linear_X", "Delta_X_X", axs, 0, 1)
plot_data(data, data_standardized, "Ground_Truth_Y", "Delta_X_Y", axs, 1, 0)
plot_data(data, data_standardized, "Velocity_Linear_Y", "Delta_X_Y", axs, 1, 1)
plot_data(data, data_standardized, "Ground_Truth_Yaw", "delta_X_Yaw", axs, 2, 0)
plot_data(data, data_standardized, "Velocity_Angular_Yaw", "delta_X_Yaw", axs, 2, 1)

plt.tight_layout()
plt.show()
