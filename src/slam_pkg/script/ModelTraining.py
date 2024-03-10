import pandas as pd
import GPy


data_df = pd.read_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv')

features = ['Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw', 'Accel_Linear_X', 'Accel_Linear_Y', 'Accel_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

X = data_df[features].values
Y = data_df[target].values

kernel = GPy.kern.RBF(input_dim=len(features), variance=1., lengthscale=1.)
model = GPy.models.GPRegression(X, Y, kernel)

model.optimize(messages=True)

print(model)
rospy.loginfo("FUCK")