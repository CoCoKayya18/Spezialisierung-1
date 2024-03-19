import pandas
import os
import pickle



datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'
tunedModelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodelTuned'

#Needs rerun
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