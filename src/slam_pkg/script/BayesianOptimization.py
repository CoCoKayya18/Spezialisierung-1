import pandas
import os
import pickle
import GPyOpt
from sklearn.metrics import mean_squared_error


# Objective function to be minimized
def objective_function(params):
    params = params[0]  # GPyOpt passes the parameters as a list of lists
    variance, length_scale = params
    
    # Update the model with new hyperparameters
    model.kern.variance = variance
    model.kern.lengthscale = length_scale
    model.optimize(messages=False)  # Optimize the model with the new kernel parameters
    
    # Prediction and evaluation
    Y_pred, _ = model.predict(X_val)
    score = mean_squared_error(Y_val, Y_pred)
    
    return score



datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'
tunedModelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodelTuned'


ith_datapoint = 1000
isSparse = ''
# isSparse = 'sparse0_'
isTuned = 'GridSearchTuned_'
# isTuned = ''

train_datafile = f'{isSparse}{ith_datapoint}_DP_train_data.csv'
val_datafile = f'{isSparse}{ith_datapoint}_DP_val_data.csv'

# Load data
train_dataframe = pandas.read_csv(os.path.join(datafilepath, train_datafile))
val_dataframe = pandas.read_csv(os.path.join(datafilepath, val_datafile))


# Load the model
model_filename = f'{isTuned}{isSparse}gpy_model_{ith_datapoint}DP.pkl'

if isTuned == '':
    with open(os.path.join(modelFilePath, model_filename), 'rb') as file:
        model = pickle.load(file)
        print(model)

if isTuned != '':
    with open(os.path.join(tunedModelFilePath, model_filename), 'rb') as file:
        model = pickle.load(file)
        print(model)

features = ['Ground_Truth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

X_train = train_dataframe[features].values
Y_train = train_dataframe[target].values
X_val = val_dataframe[features].values
Y_val = val_dataframe[target].values

# Bounds (domains) for each hyperparameter
bounds = [{'name': 'variance', 'type': 'continuous', 'domain': (1e-2, 1e2)},
          {'name': 'length_scale', 'type': 'continuous', 'domain': (1e-2, 1e2)}]

# Optimization instance creation
optimizer = GPyOpt.methods.BayesianOptimization(f=objective_function,     # Objective function
                                                domain=bounds,            # Bounds of the search space
                                                model_type='GP',          # Type of model to use
                                                acquisition_type='EI',    # Type of acquisition to use
                                                acquisition_jitter=0.01)  # Jitter to add to the acquisition function

# Running the optimization
optimizer.run_optimization(max_iter=20)

# Best found hyperparameters
print("Best hyperparameters: ", optimizer.x_opt)

# Update model with best hyperparameters
model.kern.variance = optimizer.x_opt[0]
model.kern.lengthscale = optimizer.x_opt[1]

# Optionally, re-optimize the model
model.optimize(messages=True)

# Save the tuned model
tuned_model_filename = 'BayesianOptimizationTuned_{isSparse}gpy_model_{ith_datapoint}DP.pkl'
with open(os.path.join(tunedModelFilePath, tuned_model_filename), 'wb') as file:
    pickle.dump(model, file)

print("Tuned model saved successfully with GPyOpt.")