import pandas as pd
import matplotlib.pyplot as plt

ith_datapoint = 10
# isSparse = 'sparse_'
isSparse = ''

# Load the standardized predictions and real values
filepath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/TestingData/{isSparse}{ith_datapoint}_DP_predictions_vs_real_test.csv'
data = pd.read_csv(filepath)

# Plotting real X vs. predicted X
plt.figure(figsize=(8, 8))  # Set the figure size for better visibility
plt.scatter(data['Real_X'], data['Predicted_X'], alpha=0.7)  # Plot with some transparency for overlapping points
plt.title('Real X vs. Predicted X')  # Title of the plot
plt.xlabel('Real X')  # X-axis label
plt.ylabel('Predicted X')  # Y-axis label
plt.grid(True)  # Add a grid for easier visualization
plt.plot([data['Real_X'].min(), data['Real_X'].max()], [data['Real_X'].min(), data['Real_X'].max()], 'k--')  # Add a 1:1 line for reference
plt.show()