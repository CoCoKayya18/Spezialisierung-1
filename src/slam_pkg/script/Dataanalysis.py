import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data
data_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data_Square_World_Direction.csv'
data_df = pd.read_csv(data_path)

columns_of_interest = ['world_velocity_x', 'world_velocity_y', 'angular_velocity_yaw', 'linear_acceleration_x', 'angular_acceleration_yaw', 'delta_position_x', 'delta_position_y', 'delta_yaw']

data_df = data_df[columns_of_interest]

# Display basic information about the dataset
print(data_df.info())

# Get statistical summaries of the data
print(data_df.describe())

# Check for any missing values
print(data_df.isnull().sum())

# Check for missing values
missing_values = data_df.isnull().sum()
if missing_values.any():
    print("Missing values detected:")
    print(missing_values)
    newdf = data_df.dropna()
    data_df = newdf
else:
    print("No missing values detected.")

# Check for infinite values
if np.any(np.isinf(data_df)):
    print("Infinite values detected.")
else:
    print("No infinite values detected.")

# Check data type
data_types = data_df.dtypes
print("Data types:")
print(data_types)

# Check data range
data_range = data_df.describe().loc[['min', 'max']]
print("Data range:")
print(data_range)

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(data_df)

# Check for NaN values after standardization
nan_values = np.isnan(X_standardized)
if np.any(nan_values):
    mean_X = np.mean(X_standardized, axis=0)
    std_X = np.std(X_standardized, axis=0)
    print("Mean of standardized features (X):", mean_X)
    print("Standard deviation of standardized features (X):", std_X)
    print("NaN values detected after standardization.")
else:
    mean_X = np.mean(X_standardized, axis=0)
    std_X = np.std(X_standardized, axis=0)
    print("Mean of standardized features (X):", mean_X)
    print("Standard deviation of standardized features (X):", std_X)
    print("No NaN values detected after standardization.")

data_df.to_csv(data_path)
