from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import GPy
import joblib
import pandas
import numpy
import os


datafilepath = '\Desktop\CocoMLStuff\Data'

# Define the kernel with RBF for the function and WhiteKernel for noise level
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 100))

# # Create the Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

dataframe = pandas.read_csv(os.path.join(datafilepath, 'StandardizedData.csv'))

# every_100th_point = dataframe.iloc[::100, :]

features = ['GroundTruth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

# X = every_100th_point[features].values
# Y = every_100th_point[target].values

X = dataframe[features].values
Y = dataframe[target].values

# Standardize the data

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_standardized = scaler_X.fit_transform(X)
Y_standardized = scaler_Y.fit_transform(Y) 

# Split the data into training and testing sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)


df_X_Train = pandas.DataFrame(X_train, columns=features)
df_X_Val = pandas.DataFrame(X_train, columns=features)
df_X_Test = pandas.DataFrame(X_train, columns=features)

df_Y_Train = pandas.DataFrame(Y_train, columns=target)
df_Y_Val = pandas.DataFrame(Y_train, columns=target)
df_Y_Test = pandas.DataFrame(Y_train, columns=target)

# Combine the features and targets back into dataframes for saving
train_data = pandas.concat([df_X_Train, df_Y_Train], axis=1)
val_data = pandas.concat([df_X_Val, df_Y_Val], axis=1)
test_data = pandas.concat([df_X_Test, df_Y_Test], axis=1)


# Save the datasets to CSV files
train_data.to_csv(os.path.join(datafilepath, 'train_data.csv'), index=False)
val_data.to_csv(os.path.join(datafilepath, 'val_data.csv'), index=False)
test_data.to_csv(os.path.join(datafilepath, 'test_data.csv'), index=False)

# Fit the model
gp.fit(X_train, Y_train)

# Save the model
joblib.dump(gp, '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel/GP_model.joblib')