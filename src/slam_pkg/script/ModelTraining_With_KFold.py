from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import GPy
import pandas
import os
import pickle
import numpy as np

datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'

ith_datapoint = 1
isSparse = 'sparseKFold_'
# isSparse = ''
# SpecialCase = '_OneDirection'
SpecialCase = ''
# dataName = 'Data.csv'
dataName = 'Data_ALLSet.csv'

# Get the data out of the csv
dataframe = pandas.read_csv(os.path.join(datafilepath, dataName))
features = ['linear_velocity_x', 'angular_velocity_yaw']
target = ['delta_position_x', 'delta_position_y', 'delta_yaw']
kinematic_deltas = ['kinematic_delta_x', 'kinematic_delta_y', 'kinematic_delta_yaw']

#get every i-th datapoint out of the csv
ithDataframe = dataframe[::ith_datapoint]
X = ithDataframe[features].values
Y = ithDataframe[target].values
kinematic_delta_values = ithDataframe[kinematic_deltas].values


# Standardize the data
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_standardized = scaler_X.fit_transform(X)
Y_standardized = scaler_Y.fit_transform(Y)

# Check if the data is standardized correctly
mean_X = np.mean(X_standardized, axis=0)
std_X = np.std(X_standardized, axis=0)
mean_Y = np.mean(Y_standardized, axis=0)
std_Y = np.std(Y_standardized, axis=0)
print("Mean of standardized features (X):", mean_X)
print("Standard deviation of standardized features (X):", std_X)
print("\nMean of standardized target variable (Y):", mean_Y)
print("Standard deviation of standardized target variable (Y):", std_Y)

# Number of folds
k = 10  # or any other number of folds you want

# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Lists to store results of each fold
fold_mse = []
fold_rmse = []
fold_mae = []
fold_r2 = []

# Iterate over each split
for i, (train_index, val_index) in enumerate(kf.split(X_standardized)):
    print(f"Training on fold {i+1}/{k}...")  # Print the current fold number

    # Split the data
    X_train, X_val = X_standardized[train_index], X_standardized[val_index]
    Y_train, Y_val = Y_standardized[train_index], Y_standardized[val_index]
    
    # You may skip the following line if kinematic data is not used for cross-validation
    kinematic_train, kinematic_val = kinematic_delta_values[train_index], kinematic_delta_values[val_index]

    # Define the kernel for GPy model
    kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=1.0, lengthscale=1.0) + GPy.kern.White(X_train.shape[1], variance=1.0)

    if isSparse == '':
        # Create the not sparse model
        model = GPy.models.GPRegression(X_train, Y_train, kernel)

    if isSparse != '':
        # Create the sparse model
        model = GPy.models.SparseGPRegression(X_train, Y_train, kernel)

    # Optimize Hyperparameters
    model.optimize()
    model.optimize_restarts(num_restarts=10, verbose=False)

    # Evaluate the model on the validation set
    preds, _ = model.predict(X_val)

    mse = mean_squared_error(Y_val, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_val, preds)
    r2 = r2_score(Y_val, preds)
    
    fold_mse.append(mse)
    fold_rmse.append(rmse)
    fold_mae.append(mae)
    fold_r2.append(r2)

# Calculate average performance across all folds
average_mse = np.mean(fold_mse)
average_rmse = np.mean(fold_rmse)
average_mae = np.mean(fold_mae)
average_r2 = np.mean(fold_r2)

print("Average MSE across all folds:", average_mse)
print("Average RMSE across all folds:", average_rmse)
print("Average MAE across all folds:", average_mae)
print("Average R-squared across all folds:", average_r2)