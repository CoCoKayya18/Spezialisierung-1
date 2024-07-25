import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the theta values from CSV
df = pd.read_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/odom_thetas.csv')

# Assuming the CSV has a column named 'theta' which contains the theta values
# and 'Time' which contains the timestamps

# Plot theta over time to visually inspect changes
plt.figure(figsize=(12, 6))
plt.plot(df['Time'].values, df['Theta'].values, label='Theta')
plt.xlabel('Time')
plt.ylabel('Theta (radians)')
plt.title('Theta over Time')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the derivative of theta to find rate of change
# The derivative will peak at points where theta changes abruptly
df['theta_derivative'] = np.gradient(df['Theta'].values, df['Time'].values)

# Plot the derivative over time
plt.figure(figsize=(12, 6))
plt.plot(df['Time'].values, df['theta_derivative'].values, label='Theta Derivative')
plt.xlabel('Time')
plt.ylabel('Derivative of Theta (radians/s)')
plt.title('Derivative of Theta over Time')
plt.legend()
plt.grid(True)
plt.show()

# Find the time points where the absolute derivative exceeds a certain threshold
# which may correspond to corners if the change is abrupt
threshold = 0.5  # Set a threshold for the derivative that indicates a sharp turn
sharp_turns = df['Time'][np.abs(df['theta_derivative']) > threshold]

print("Sharp turns likely occurred at these timestamps:")
print(sharp_turns)
