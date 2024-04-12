import pandas as pd
import matplotlib.pyplot as plt
import os

ith_datapoint = 1
# isSparse = 'sparse40k_'
isSparse = ''
SpecialCase = '_OneDirection'
# SpecialCase = ''
# dataName = 'Data.csv'
TestOrTrain = '_train'
TestOrTrain = '_test'
isTuned = ''

# Load your DataFrame
filename = f'{isTuned}{isSparse}{ith_datapoint}{SpecialCase}_DP_predictions_vs_real_train.csv'
DataFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'

# DataFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/sparseOnlyX_1_DP_predictions_vs_real_test_OnlyX.csv'

print(os.path.join(DataFilePath, filename))
df = pd.read_csv(os.path.join(DataFilePath, filename))

getKinematic = False
getGT = True
getPredicted = True

limit = False

x_limit = (-0.1, 0.1)
y_limit = (-0.1, 0.1)

plt.figure(figsize=(12, 6))

if getKinematic:
    df['cumulative_kinematic_x'] = df['Kinematic_X'].cumsum()
    df['cumulative_kinematic_y'] = df['Kinematic_Y'].cumsum()
    # df['cumulative_kinematic_x'] = df['kinematic_delta_x'].cumsum()
    # df['cumulative_kinematic_y'] = df['kinematic_delta_y'].cumsum()
    cumulative_kinematic_x = df['cumulative_kinematic_x'].to_numpy()
    cumulative_kinematic_y = df['cumulative_kinematic_y'].to_numpy()
    plt.plot(cumulative_kinematic_x, cumulative_kinematic_y, label='Kinematic Path', marker='o', linestyle='-', markersize=2)

if getGT: 
    df['cumulative_ground_truth_x'] = df['Real_X'].cumsum()
    print(df['Real_X'].cumsum())
    df['cumulative_ground_truth_y'] = df['Real_Y'].cumsum()
    print(df['Real_Y'].cumsum())
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
