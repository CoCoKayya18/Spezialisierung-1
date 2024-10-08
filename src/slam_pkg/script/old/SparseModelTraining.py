import GPy
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas

datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'

ith_datapoint = 1
sparse_number = 0
isSparse = 'sparse'
# isSparse = ''

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
df_X_Train = pandas.DataFrame(X_train, columns=features)
df_X_Val = pandas.DataFrame(X_val, columns=features)
df_X_Test = pandas.DataFrame(X_test, columns=features)
df_Y_Train = pandas.DataFrame(Y_train, columns=target)
df_Y_Val = pandas.DataFrame(Y_val, columns=target)
df_Y_Test = pandas.DataFrame(Y_test, columns=target)
train_data = pandas.concat([df_X_Train, df_Y_Train], axis=1)
val_data = pandas.concat([df_X_Val, df_Y_Val], axis=1)
test_data = pandas.concat([df_X_Test, df_Y_Test], axis=1)
train_data.to_csv(os.path.join(datafilepath, f'{isSparse}{ith_datapoint}_DP_train_data.csv'), index=False)
val_data.to_csv(os.path.join(datafilepath, f'{isSparse}{ith_datapoint}_DP_val_data.csv'), index=False)
test_data.to_csv(os.path.join(datafilepath, f'{isSparse}{ith_datapoint}_DP_test_data.csv'), index=False)

# Define the kernel with RBF for the function and WhiteKernel for noise level
kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=1.0, lengthscale=1.0) + GPy.kern.White(X_train.shape[1], variance=1.0)

# Create the sparse model
sparse_model = GPy.models.SparseGPRegression(X_train, Y_train, kernel)

# Optimize Hyperparameters
sparse_model.optimize()
sparse_model.optimize_restarts(num_restarts=10, verbose=True)

print(sparse_model)

# Save the model and scalers
model_filename = f'sparse{sparse_number}_gpy_model_{ith_datapoint}DP.pkl'
scaler_filenameX = f'sparse_scaler_X_{ith_datapoint}.pkl'
scaler_filenameY = f'sparse_scaler_Y_{ith_datapoint}.pkl'

with open(os.path.join(modelFilePath, model_filename), 'wb') as file:
    pickle.dump(sparse_model, file)

with open(os.path.join(scalerFilePath, scaler_filenameX), 'wb') as file:
    pickle.dump(scaler_X, file)

with open(os.path.join(scalerFilePath, scaler_filenameY), 'wb') as file:
    pickle.dump(scaler_Y, file)
