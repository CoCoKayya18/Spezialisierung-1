import pandas as pd
import matplotlib.pyplot as plt

# TestOrTrain = '_test'
TestOrTrain = '_train'

# Load your DataFrame
DataFilePath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/sparse0_1_DP_predictions_vs_real{TestOrTrain}.csv'

df = pd.read_csv(DataFilePath)

getKinematic = True
getGT = True
getPredicted = True

limit = False

x_limit = (-0.00, 0.1)
y_limit = (-0.0, 0.1)

plt.figure(figsize=(12, 6))

if getKinematic:
    df['cumulative_kinematic_x'] = df['Kinematic_X'].cumsum()
    df['cumulative_kinematic_y'] = df['Kinematic_Y'].cumsum()
    cumulative_kinematic_x = df['cumulative_kinematic_x'].to_numpy()
    cumulative_kinematic_y = df['cumulative_kinematic_y'].to_numpy()
    plt.plot(cumulative_kinematic_x, cumulative_kinematic_y, label='Kinematic Path', marker='o', linestyle='-', markersize=2)

if getGT: 
    df['cumulative_ground_truth_x'] = df['Real_X'].cumsum()
    df['cumulative_ground_truth_y'] = df['Real_Y'].cumsum()
    cumulative_ground_truth_x = df['cumulative_ground_truth_x'].to_numpy()
    cumulative_ground_truth_y = df['cumulative_ground_truth_y'].to_numpy()
    plt.plot(cumulative_ground_truth_x, cumulative_ground_truth_y, label='Ground Truth Path', marker='x', linestyle='--', markersize=2)

if getPredicted:
    df['cumulative_predicted_x'] = df['Predicted_X'].cumsum()
    df['cumulative_predicted_y'] = df['Predicted_Y'].cumsum()
    cumulative_predicted_x = df['cumulative_predicted_x'].to_numpy()
    cumulative_predicted_y = df['cumulative_predicted_y'].to_numpy()
    plt.plot(cumulative_predicted_x, cumulative_predicted_y, label='Predicted Path', marker='x', linestyle='dotted', markersize=2)

if limit:
    plt.xlim(x_limit)
    plt.ylim(y_limit)

# Adding titles and labels
plt.title('Robot Trajectories')
plt.xlabel('Cumulative Delta X')
plt.ylabel('Cumulative Delta Y')

# Adding a legend
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()
