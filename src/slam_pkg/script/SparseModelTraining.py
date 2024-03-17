import GPy
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'

ith_datapoint = 1

# Load data
dataframe = pd.read_csv(os.path.join(datafilepath, 'Data.csv'))
features = ['Ground_Truth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

# Get every i-th datapoint
ithDataframe = dataframe[::ith_datapoint]
X = ithDataframe[features].values
Y = ithDataframe[target].values

# Standardize the data
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_standardized = scaler_X.fit_transform(X)
Y_standardized = scaler_Y.fit_transform(Y) 

# Split the data into training, validation, and testing sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X_standardized, Y_standardized, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Define the kernel with RBF for the function and WhiteKernel for noise level
kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=1.0, lengthscale=1.0) + GPy.kern.White(X_train.shape[1], variance=1.0)

# Create the sparse model
sparse_model = GPy.models.SparseGPRegression(X_train, Y_train, kernel)

# Optimize Hyperparameters
sparse_model.optimize()
sparse_model.optimize_restarts(num_restarts=10, verbose=True)

print(sparse_model)

# Save the model and scalers
model_filename = f'sparse_gpy_model_{ith_datapoint}DP.pkl'
scaler_filenameX = f'sparse_scaler_X_{ith_datapoint}.pkl'
scaler_filenameY = f'sparse_scaler_Y_{ith_datapoint}.pkl'

with open(os.path.join(modelFilePath, model_filename), 'wb') as file:
    pickle.dump(sparse_model, file)

with open(os.path.join(scalerFilePath, scaler_filenameX), 'wb') as file:
    pickle.dump(scaler_X, file)

with open(os.path.join(scalerFilePath, scaler_filenameY), 'wb') as file:
    pickle.dump(scaler_Y, file)
