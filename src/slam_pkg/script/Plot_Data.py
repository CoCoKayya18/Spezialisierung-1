import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
file_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Odometry_Position plot
axs[0].plot(data.index.to_numpy(), data["Odometry_Position"].to_numpy(), label="Odometry_Position")
axs[0].set_title("Odometry Position Over Time")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Position")
axs[0].grid(True)
axs[0].legend()

# Odometry_Velocity plot
axs[1].plot(data.index.to_numpy(), data["Odometry_Velocity"].to_numpy(), label="Odometry_Velocity", color="red")
axs[1].set_title("Odometry Velocity Over Time")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Velocity")
axs[1].grid(True)
axs[1].legend()

# Delta_X plot
axs[2].plot(data.index.to_numpy(), data["Delta_X"].to_numpy(), label="Delta_X", color="green")
axs[2].set_title("Delta X Over Time")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Delta X")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()
