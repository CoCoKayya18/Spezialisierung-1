import GPy
import pandas
import numpy
import os
import pickle
import matplotlib.pyplot as plt




# Use the validation set to tune the hyperparameters
def validate_model(model, X_val, Y_val):
    # Make predictions on the validation set
    Y_pred, _ = model.predict(X_val)
    # Calculate the mean squared error or any other appropriate metric
    mse = numpy.mean((Y_pred - Y_val)**2)
    return mse



datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'
tunedModelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodelTuned'

ith_datapoint = 20
isSparse = ''
# isSparse = 'sparse_1_'

train_datafile = f'{isSparse}{ith_datapoint}_DP_train_data.csv'
val_datafile = f'{isSparse}{ith_datapoint}_DP_val_data.csv'

# Load data
train_dataframe = pandas.read_csv(os.path.join(datafilepath, train_datafile))
val_dataframe = pandas.read_csv(os.path.join(datafilepath, val_datafile))


# Load the model
model_filename = f'{isSparse}gpy_model_{ith_datapoint}DP.pkl'
with open(os.path.join(modelFilePath, model_filename), 'rb') as file:
    model = pickle.load(file)
    print(model)

features = ['Ground_Truth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

X_train = train_dataframe[features].values
Y_train = train_dataframe[target].values
X_val = val_dataframe[features].values
Y_val = val_dataframe[target].values

best_mse = float('inf')
best_model = None
best_kernel_params = None

variances, lengthscales, mses = [], [], []

variance_range = numpy.logspace(-2, 2, 10)
lengthscale_range = numpy.logspace(-2, 2, 10)

# Loop over hyperparameters to tune
for var in variance_range:
    for ls in lengthscale_range:
        # Create a new model or copy the initial model
        kernel = GPy.kern.RBF(input_dim=len(features), variance=var, lengthscale=ls)
        model = GPy.models.GPRegression(X_train, Y_train, kernel)
        
        # Optimize model on training data
        model.optimize(messages=False)
        model.optimize_restarts(num_restarts = 10, verbose=True)
        
        # Evaluate the model on the validation set
        mse = validate_model(model, X_val, Y_val)
        print(f"Variance: {var}, Lengthscale: {ls}, MSE: {mse}")
        
        # Keep track of the best model
        if mse < best_mse:
            best_mse = mse
            best_model = model.copy()
            best_kernel_params = {'variance': var, 'lengthscale': ls}
        
        variances.append(var)
        lengthscales.append(ls)
        mses.append(mse)

print(f"Best MSE: {best_mse}")
print(f"Best Kernel Parameters: {best_kernel_params}")

tuned_model_filename = f'GridSearchTuned_{isSparse}gpy_model_{ith_datapoint}DP.pkl'

with open(os.path.join(tunedModelFilePath, tuned_model_filename), 'wb') as file:
    pickle.dump(model, file)

# Plotting the results
fig = plt.figure(figsize=(14, 6))

# MSE vs Variance
ax1 = fig.add_subplot(121)
sc = ax1.scatter(variances, mses, c=lengthscales, cmap='viridis')
ax1.set_xlabel('Kernel Variance')
ax1.set_ylabel('MSE')
ax1.set_title('MSE vs Kernel Variance')
plt.colorbar(sc, label='Lengthscale')

# MSE vs Lengthscale
ax2 = fig.add_subplot(122)
sc = ax2.scatter(lengthscales, mses, c=variances, cmap='viridis')
ax2.set_xlabel('Lengthscale')
ax2.set_ylabel('MSE')
ax2.set_title('MSE vs Lengthscale')
plt.colorbar(sc, label='Kernel Variance')

plt.tight_layout()
plt.show()
