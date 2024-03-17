import GPy
import pandas
import numpy
import os
import pickle

datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'

ith_datapoint = 100
isSparse = ''
# isSparse = 'sparse_1_'

datafile = f'{isSparse}{ith_datapoint}_DP_val_data.csv'

# Load data
dataframe = pandas.read_csv(os.path.join(datafilepath, datafile))

# Load model and Scaler
model_filename = f'{isSparse}gpy_model_{ith_datapoint}DP.pkl'
# scaler_filenameX = f'{isSparse}scaler_X_{ith_datapoint}.pkl'
# scaler_filenameY = f'{isSparse}scaler_Y_{ith_datapoint}.pkl'

# Load the model
with open(os.path.join(modelFilePath, model_filename), 'rb') as file:
    model = pickle.load(file)
    print(model)

features = ['Ground_Truth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

X_val = dataframe[features].values
Y_val = dataframe[target].values

# Use the validation set to tune the hyperparameters
def validate_model(model, X_val, Y_val):
    # Make predictions on the validation set
    Y_pred, _ = model.predict(X_val)
    # Calculate the mean squared error or any other appropriate metric
    mse = numpy.mean((Y_pred - Y_val)**2)
    return mse

best_mse = float('inf')
best_model = None
best_kernel_params = None

# Try different hyperparameter settings
for var in numpy.logspace(-1, 1, 10):  # Example range for variance
    for ls in numpy.logspace(-1, 1, 10):  # Example range for lengthscale
        model.kern.rbf.variance = var
        model.kern.rbf.lengthscale = ls
        # Optimize the model hyperparameters
        model.optimize(messages=False)
        # Evaluate the model on the validation set
        mse = validate_model(model, X_val, Y_val)
        print(f"Variance: {var}, Lengthscale: {ls}, MSE: {mse}")
        # Keep track of the best model
        if mse < best_mse:
            best_mse = mse
            best_model = model.copy()
            best_kernel_params = (var, ls)

print(f"Best MSE: {best_mse}")
print(f"Best Kernel Parameters: {best_kernel_params}")
