from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import joblib
import pandas
import numpy

# Assuming X and Y are your training data and targets, respectively.
# You may want to standardize them before training.

# Define the kernel with RBF for the function and WhiteKernel for noise level
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 100))

# # Create the Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# dataframe = pandas.read_csv('/content/Data.csv')

# features = ['GroundTruth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
# target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

# X = dataframe[features].values
# Y = dataframe[target].values

# # Fit the model
# gp.fit(X, Y)

# Save the model
joblib.dump(gp, '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel/GP_model.joblib')