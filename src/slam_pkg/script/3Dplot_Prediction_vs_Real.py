import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ith_datapoint = 1
isSparse = 'sparseKFoldSquareRobotFrameDirection_'
# isSparse = ''
SpecialCase = '_Square_RobotFrameDeltas_Direction'
# SpecialCase = ''
# dataName = 'Data.csv'
dataName = 'Data_Square_RobotFrameDeltas_Direction'
# isTuned = 'BayesianOptimizationTuned_'
# isTuned = 'GridSearchTuned_'
# isTuned = 'BayesianOptimizationTuned_GridSearchTuned_'
isTuned = ''
# trainOrTest = '_train'
trainOrTest = '_test'

# Load predictions and real values
filepath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isTuned}{isSparse}{ith_datapoint}{SpecialCase}_DP_predictions_vs_real{trainOrTest}.csv'

data = pd.read_csv(filepath)

features = ['X', 'Y', 'Yaw']
predicted_prefix = 'Predicted_'
real_prefix = 'Real_'

# Calculate the Mean Squared Error (MSE) for each feature
mse_X = np.mean((data['Predicted_X'] - data['Real_X']) ** 2)
mse_Y = np.mean((data['Predicted_Y'] - data['Real_Y']) ** 2)
mse_Yaw = np.mean((data['Predicted_Yaw'] - data['Real_Yaw']) ** 2)

# Overall MSE (considering all features together)
mse_overall = np.mean((data[['Predicted_X', 'Predicted_Y', 'Predicted_Yaw']].values - 
                       data[['Real_X', 'Real_Y', 'Real_Yaw']].values) ** 2)

fig = plt.figure(figsize=(14, 7))

# Actual vs. Predicted subplot
ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 columns, 1st subplot
ax1.scatter(data[f'{real_prefix}{features[0]}'], data[f'{real_prefix}{features[1]}'], data[f'{real_prefix}{features[2]}'], c='blue', marker='o', label='Real Values')
ax1.scatter(data[f'{predicted_prefix}{features[0]}'], data[f'{predicted_prefix}{features[1]}'], data[f'{predicted_prefix}{features[2]}'], c='red', marker='^', label='Predicted Values')
ax1.set_xlabel(f'{features[0]} Delta')
ax1.set_ylabel(f'{features[1]} Delta')
ax1.set_zlabel(f'{features[2]} Delta')
ax1.set_title('Real vs. Predicted Values')
ax1.legend()

# Error subplot
ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, 2nd subplot
errors = np.sqrt((data[f'{predicted_prefix}{features[0]}'] - data[f'{real_prefix}{features[0]}'])**2 +
                 (data[f'{predicted_prefix}{features[1]}'] - data[f'{real_prefix}{features[1]}'])**2 +
                 (data[f'{predicted_prefix}{features[2]}'] - data[f'{real_prefix}{features[2]}'])**2)
sc = ax2.scatter(data[f'{real_prefix}{features[0]}'], data[f'{real_prefix}{features[1]}'], data[f'{real_prefix}{features[2]}'], c=errors, cmap='viridis', label='Error Magnitude')
ax2.set_xlabel(f'{features[0]} Delta')
ax2.set_ylabel(f'{features[1]} Delta')
ax2.set_zlabel(f'{features[2]}')
ax2.set_title('Error Magnitude (Euclidean distance in between)')
ax2.legend()

cbar = fig.colorbar(sc, ax=ax2)
cbar.set_label('Error magnitude')

# Show MSE error
plt.figtext(0.5, 0.01, f'Overall MSE: {mse_overall:.4f} | MSE in X: {mse_X:.4f} | MSE in Y: {mse_Y:.4f} | MSE in Yaw: {mse_Yaw:.4f}', 
            ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

# Show the plots
plt.tight_layout()
plt.show()
