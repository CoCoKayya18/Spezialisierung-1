import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'
data_df = pd.read_csv(data_path)

# Display basic information about the dataset
print(data_df.info())

# Get statistical summaries of the data
print(data_df.describe())

# Check for any missing values
print(data_df.isnull().sum())

# # Plot the distribution of Yaw values
# sns.histplot(data_df['Ground_Truth_Yaw'], kde=True)
# plt.show()



# data_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/StandardizedData.csv'
# data_df = pd.read_csv(data_path)

# # Display basic information about the dataset
# print(data_df.info())

# # Get statistical summaries of the data
# print(data_df.describe())

# # Check for any missing values
# print(data_df.isnull().sum())
