from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import GPy
import pandas
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_distribution(X_train, X_test, feature_names, title_suffix=''):
    num_features = len(feature_names)
    fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 4))
    
    if num_features == 1:
        axes = [axes]
    
    for i, feature_name in enumerate(feature_names):
        axes[i].hist(X_train[:, i], bins=20, color='blue', alpha=0.5, label='Train')
        axes[i].hist(X_test[:, i], bins=20, color='orange', alpha=0.5, label='Test')
        axes[i].set_title(f'{feature_name} Distribution {title_suffix}')
        axes[i].set_xlabel(feature_name)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'

ith_datapoint = 1
isSparse = 'sparseKFoldSquareWorldDirection_'
# isSparse = ''
SpecialCase = '_Square_World_Direction'
# SpecialCase = ''
# dataName = 'Data.csv'
dataName = 'Data_Square_World_Direction.csv'

# Get the data out of the csv
dataframe = pandas.read_csv(os.path.join(datafilepath, dataName))
# features = ['linear_velocity_x', 'angular_velocity_yaw']
features = ['world_velocity_x', 'world_velocity_y', 'angular_velocity_yaw']
target = ['delta_position_x', 'delta_position_y', 'delta_yaw']
kinematic_deltas = ['kinematic_delta_x', 'kinematic_delta_y', 'kinematic_delta_yaw']

#get every i-th datapoint out of the csv
ithDataframe = dataframe[::ith_datapoint]
X = ithDataframe[features].values
Y = ithDataframe[target].values
kinematic_data = dataframe[kinematic_deltas].values


# # Standardize the data
# scaler_X = StandardScaler()
# scaler_Y = StandardScaler()
# X_standardized = scaler_X.fit_transform(X)
# Y_standardized = scaler_Y.fit_transform(Y)

# Check if the data is standardized correctly

X_train_full, X_test, Y_train_full, Y_test, kinematic_train_full, kinematic_test = train_test_split(X, Y, kinematic_data, test_size=0.3, random_state=42)

# Plotting target distributions
plot_feature_distribution(X_train_full, X_test, features, title_suffix='(Features)')
plot_feature_distribution(Y_train_full, Y_test, target, title_suffix='(Targets)')

# Then fit and transform with standardization
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_full = scaler_X.fit_transform(X_train_full)
X_test = scaler_X.transform(X_test)  # Only transform the test data
Y_train_full = scaler_Y.fit_transform(Y_train_full)
Y_test = scaler_Y.transform(Y_test)  # Only transform the test data

mean_X = np.mean(X_train_full, axis=0)
std_X = np.std(X_train_full, axis=0)
mean_Y = np.mean(Y_train_full, axis=0)
std_Y = np.std(Y_train_full, axis=0)
print("Mean of standardized features (X):", mean_X)
print("Standard deviation of standardized features (X):", std_X)
print("Mean of standardized target variable (Y):", mean_Y)
print("Standard deviation of standardized target variable (Y):", std_Y)

mean_X = np.mean(X_test, axis=0)
std_X = np.std(X_test, axis=0)
mean_Y = np.mean(Y_test, axis=0)
std_Y = np.std(Y_test, axis=0)
print("\nMean of standardized features (X):", mean_X)
print("Standard deviation of standardized features (X):", std_X)
print("Mean of standardized target variable (Y):", mean_Y)
print("Standard deviation of standardized target variable (Y):", std_Y)

# Number of folds
k = 5  # or any other number of folds you want

# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

best_r2 = float('-inf')  # Initialize to negative infinity
best_model = None
best_indices = None

all_X_train = []
all_Y_train = []
all_X_val = []
all_Y_val = []

# Lists to store results of each fold
fold_mse = []
fold_rmse = []
fold_mae = []
fold_r2 = []

# Iterate over each split
for i, (train_index, val_index) in enumerate(kf.split(X_train_full)):

    print(f"Training on fold {i+1}/{k}...")  # Print the current fold number

    # Split the data
    X_train, X_val = X_train_full[train_index], X_train_full[val_index]
    Y_train, Y_val = Y_train_full[train_index], Y_train_full[val_index]
    kinematic_train, kinematic_val = kinematic_train_full[train_index], kinematic_train_full[val_index]

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

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_indices = (train_index, val_index)
    
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

print("MSE across all folds:", fold_mse)
print("RMSE across all folds:", fold_rmse)
print("MAE across all folds:", fold_mae)
print("R-squared across all folds:", fold_r2)

print(best_model)

# Save the model and scalers
model_filename = f'{isSparse}gpy_model_{ith_datapoint}DP{SpecialCase}.pkl'
scaler_filenameX = f'{isSparse}scaler_X_{ith_datapoint}{SpecialCase}.pkl'
scaler_filenameY = f'{isSparse}scaler_Y_{ith_datapoint}{SpecialCase}.pkl'

with open(os.path.join(modelFilePath, model_filename), 'wb') as file:
    pickle.dump(best_model, file)

with open(os.path.join(scalerFilePath, scaler_filenameX), 'wb') as file:
    pickle.dump(scaler_X, file)

with open(os.path.join(scalerFilePath, scaler_filenameY), 'wb') as file:
    pickle.dump(scaler_Y, file)

# Prepare datasets for the best model
X_train, X_val = X_train_full[best_indices[0]], X_train_full[best_indices[1]]
Y_train, Y_val = Y_train_full[best_indices[0]], Y_train_full[best_indices[1]]
kinematic_train, kinematic_val = kinematic_train_full[best_indices[0]], kinematic_train_full[best_indices[1]]

# Inverse transform features and targets for saving to CSV
X_train_inv = scaler_X.inverse_transform(X_train)
X_val_inv = scaler_X.inverse_transform(X_val)
X_test_inv = scaler_X.inverse_transform(X_test)

Y_train_inv = scaler_Y.inverse_transform(Y_train)
Y_val_inv = scaler_Y.inverse_transform(Y_val)
Y_test_inv = scaler_Y.inverse_transform(Y_test)

# Save data for the best model
# Concatenate the inversely transformed data and kinematic data
train_data = pandas.concat([pandas.DataFrame(X_train_inv, columns=features), pandas.DataFrame(Y_train_inv, columns=target), pandas.DataFrame(kinematic_train, columns=kinematic_deltas)], axis=1)

val_data = pandas.concat([pandas.DataFrame(X_val_inv, columns=features), pandas.DataFrame(Y_val_inv, columns=target), pandas.DataFrame(kinematic_val, columns=kinematic_deltas)], axis=1)

test_data = pandas.concat([pandas.DataFrame(X_test_inv, columns=features), pandas.DataFrame(Y_test_inv, columns=target), pandas.DataFrame(kinematic_test, columns=kinematic_deltas)], axis=1)

train_data.to_csv(os.path.join(datafilepath, f'{isSparse}{ith_datapoint}_DP_train_data{SpecialCase}.csv'), index=False)
val_data.to_csv(os.path.join(datafilepath, f'{isSparse}{ith_datapoint}_DP_val_data{SpecialCase}.csv'), index=False)
test_data.to_csv(os.path.join(datafilepath, f'{isSparse}{ith_datapoint}_DP_test_data{SpecialCase}.csv'), index=False)