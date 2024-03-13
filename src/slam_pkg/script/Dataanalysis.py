import pandas as pd

# Load the data
data_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'
data_df = pd.read_csv(data_path)

# Display basic information about the dataset
print(data_df.info())

# Get statistical summaries of the data
print(data_df.describe())

# Check for any missing values
print(data_df.isnull().sum())

data_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/StandardizedData.csv'
data_df = pd.read_csv(data_path)

# Display basic information about the dataset
print(data_df.info())

# Get statistical summaries of the data
print(data_df.describe())

# Check for any missing values
print(data_df.isnull().sum())
