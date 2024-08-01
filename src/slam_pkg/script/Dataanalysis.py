import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore
from tqdm import tqdm

def get_comb_paths(base_dir, suffix=''):
    comb_paths = []
    for root, dirs, files in os.walk(base_dir):
        if f'FullData{suffix}.csv' in files:
            comb_paths.append((root, f'FullData{suffix}.csv'))
    return comb_paths

def process_and_check_data(file_path):
    try:
        data_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    
    columns_of_interest = ['Theta_calculated', 'yaw_world', 'linear_velocity_x', 
                       'world_velocity_x', 'world_velocity_y', 'angular_velocity_yaw', 
                       'linear_acceleration_x', 'angular_acceleration_yaw', 
                       'delta_position_x_world', 'delta_position_y_world', 'delta_yaw', 
                       'kinematic_delta_x', 'kinematic_delta_y', 'kinematic_delta_yaw', 'twist.twist.linear.x', 'Time']
    
    if not all(col in data_df.columns for col in columns_of_interest):
        print(f"Columns of interest not found in {file_path}")
        return
    
    data_df = data_df[columns_of_interest]

    report = []
    
    # Basic information
    report.append("Basic Information:")
    report.append(str(data_df.info()))
    report.append("\n")

    # Data length
    report.append("Dataset Initial Length:")
    report.append(str(len(data_df)))
    report.append("\n")
    
    # Statistical summaries
    report.append("Statistical Summaries:")
    report.append(str(data_df.describe()))
    report.append("\n")
    
    # Missing values
    missing_values = data_df.isnull().sum()
    if missing_values.any():
        report.append("Missing values detected:")
        report.append(str(missing_values))
        data_df = data_df.dropna()
    else:
        report.append("No missing values detected.")
    report.append("\n")
    
    # Infinite values
    if np.any(np.isinf(data_df)):
        report.append("Infinite values detected.")
        data_df = data_df.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        report.append("No infinite values detected.")
    report.append("\n")
    
    # Data types
    report.append("Data Types:")
    report.append(str(data_df.dtypes))
    report.append("\n")
    
    # Data range
    data_range = data_df.describe().loc[['min', 'max']]
    report.append("Data Range:")
    report.append(str(data_range))
    report.append("\n")

    # Duplicate rows
    duplicate_rows = data_df.duplicated().sum()
    if duplicate_rows > 0:
        report.append(f"Duplicate rows detected: {duplicate_rows}")
        data_df = data_df.drop_duplicates()
    else:
        report.append("No duplicate rows detected.")
    report.append("\n")

    # Remove near-duplicates
    # threshold = 1e-4  # Define a threshold for considering values as near-duplicates
    # near_duplicate_rows = 0

    # # Create a mask to identify near-duplicate rows
    # mask = pd.Series([False] * len(data_df))

    # for i in range(len(data_df) - 1):
    #     row = data_df.iloc[i]
    #     for j in range(i + 1, len(data_df)):
    #         if all((data_df.iloc[j] - row).abs() < threshold):
    #             mask[j] = True

    # # Count and remove near-duplicate rows
    # near_duplicate_rows = mask.sum()
    # if near_duplicate_rows > 0:
    #     report.append(f"Near-duplicate rows detected and removed: {near_duplicate_rows}")
    #     data_df = data_df[~mask]
    # else:
    #     report.append("No near-duplicate rows detected.")
    # report.append("\n")

    # Outlier detection using Z-score
    z_scores = np.abs(zscore(data_df))
    outliers = np.where(z_scores > 3)
    if len(outliers[0]) > 0:
        report.append(f"Outliers detected: {len(outliers[0])}")
        data_df = data_df[(z_scores < 3).all(axis=1)]
    else:
        report.append("No outliers detected.")
    report.append("\n")

    # Correlation matrix
    report.append("Correlation Matrix:")
    correlation_matrix = data_df.corr()
    report.append(str(correlation_matrix))
    report.append("\n")

    # Inspect unique values and their counts
    report.append("\nQuantization Check:")

    unique_values_x = data_df['delta_position_x_world'].value_counts().sort_index()
    unique_values_y = data_df['delta_position_y_world'].value_counts().sort_index()
    unique_values_yaw = data_df['delta_yaw'].value_counts().sort_index()

    report.append("Unique Values x: ")
    report.append(str(unique_values_x))
    report.append("Unique Values y: ")
    report.append(str(unique_values_y))
    report.append("Unique Values yaw: ")
    report.append(str(unique_values_yaw))
    
    # Standardize the data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(data_df)
    
    # Check for NaN values after standardization
    nan_values = np.isnan(X_standardized)
    if np.any(nan_values):
        mean_X = np.mean(X_standardized, axis=0)
        std_X = np.std(X_standardized, axis=0)
        report.append("NaN values detected after standardization.")
        report.append(f"Mean of standardized features (X): {mean_X}")
        report.append(f"Standard deviation of standardized features (X): {std_X}")
    else:
        mean_X = np.mean(X_standardized, axis=0)
        std_X = np.std(X_standardized, axis=0)
        report.append("\nNo NaN values detected after standardization.")
        report.append(f"Mean of standardized features (X): {mean_X}")
        report.append(f"Standard deviation of standardized features (X): {std_X}")

    # Calculate error metrics between 'yaw_world' and 'Theta_calculated'
    mae = mean_absolute_error(data_df['yaw_world'].values, data_df['Theta_calculated'].values)
    rmse = np.sqrt(mean_squared_error(data_df['yaw_world'].values, data_df['Theta_calculated'].values))
    report.append(f"\nMean Absolute Error (MAE) between yaw_world and Theta_calculated: {mae}")
    report.append(f"Root Mean Squared Error (RMSE) between yaw_world and Theta_calculated: {rmse}")
    report.append("\n")

    # New Data length
    report.append("\nDataset New Length:")
    report.append(str(len(data_df)))
    report.append("\n")
    
    # Save the report to a text file in the same directory
    report_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace('.csv', '_report.txt'))
    with open(report_path, 'w') as f:
        for line in report:
            f.write(line + "\n")
    
    # Save cleaned data to a new CSV file with '_cleaned' suffix
    cleaned_data_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace('.csv', '_cleaned.csv'))
    data_df.to_csv(cleaned_data_path, index=False)

    print(f"Report saved to {report_path}")
    print(f"Cleaned data saved to {cleaned_data_path}")

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(file_path), "plots")
    histograms_dir = os.path.join(plots_dir, "histograms")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(histograms_dir):
        os.makedirs(histograms_dir)

    # Generate and save plots
    sns.set(style="whitegrid")

    # Histograms
    for col in data_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data_df[col].values, kde=True)
        plt.title(f'Histogram of {col}')
        plt.savefig(os.path.join(histograms_dir, f'{col}_histogram.png'))
        plt.close()

    # Pair plot
    sns.pairplot(data_df)
    plt.savefig(os.path.join(plots_dir, 'pairplot.png'))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()

    # Plot yaw_world vs theta_calculated with error metrics
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['yaw_world'].values, label='yaw_world')
    plt.plot(data_df['Theta_calculated'].values, label='theta_calculated')
    plt.xlabel('Index')
    plt.ylabel('Angle (radians)')
    plt.title(f'Yaw World vs Theta Calculated\nMAE: {mae:.4f}, RMSE: {rmse:.4f}')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'yaw_vs_theta.png'))
    plt.close()

    # Plot ground truth vs kinematic deltas path
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(data_df['delta_position_x_world'].values), np.cumsum(data_df['delta_position_y_world'].values), label='Ground Truth Path')
    plt.plot(np.cumsum(data_df['kinematic_delta_x'].values), np.cumsum(data_df['kinematic_delta_y'].values), label='Kinematic Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Ground Truth Path vs Kinematic Path')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'ground_truth_vs_kinematic_path.png'))
    plt.close()

    # Plot linear_velocity_x vs odom_linear_velocity_x
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['linear_velocity_x'].values, label='linear_velocity_x')
    plt.plot(data_df['twist.twist.linear.x'].values, label='odom_linear_velocity_x')
    plt.xlabel('Index')
    plt.ylabel('Linear Velocity X')
    plt.title('Linear Velocity X vs Odometry Linear Velocity X')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'linear_velocity_x_vs_odom.png'))
    plt.close()

    # Convert odom velocities to world frame and plot
    cos_yaw = np.cos(data_df['yaw_world'].values)
    sin_yaw = np.sin(data_df['yaw_world'].values)
    odom_world_velocity_x = data_df['twist.twist.linear.x'] * cos_yaw
    odom_world_velocity_y = data_df['twist.twist.linear.x'] * sin_yaw

    mae_x = mean_absolute_error(data_df['world_velocity_x'], odom_world_velocity_x)
    rmse_x = np.sqrt(mean_squared_error(data_df['world_velocity_x'], odom_world_velocity_x))
    
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['Time'].values, data_df['world_velocity_x'].values, label='world_velocity_x')
    plt.plot(data_df['Time'].values, odom_world_velocity_x, label='odom_world_velocity_x')
    plt.xlabel('Time')
    plt.ylabel('World Velocity X')
    plt.title(f'World Velocity X vs Odometry World Velocity X\nMAE: {mae_x:.4f}, RMSE: {rmse_x:.4f}')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'world_velocity_x_vs_odom.png'))
    plt.close()

    mae_y = mean_absolute_error(data_df['world_velocity_y'], odom_world_velocity_y)
    rmse_y = np.sqrt(mean_squared_error(data_df['world_velocity_y'], odom_world_velocity_y))

    plt.figure(figsize=(10, 6))
    plt.plot(data_df['Time'].values, data_df['world_velocity_y'].values, label='world_velocity_y')
    plt.plot(data_df['Time'].values, odom_world_velocity_y, label='odom_world_velocity_y')
    plt.xlabel('Time')
    plt.ylabel('World Velocity Y')
    plt.title(f'World Velocity Y vs Odometry World Velocity Y\nMAE: {mae_y:.4f}, RMSE: {rmse_y:.4f}')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'world_velocity_y_vs_odom.png'))
    plt.close()

    print(f"Plots saved to {plots_dir}")


if __name__ == '__main__':
    base_dir = '../Spezialisierung-1/src/slam_pkg/data'
    
    # Get paths for FullData and FullData_single
    combPaths = get_comb_paths(base_dir)
    combPaths_single = get_comb_paths(base_dir, suffix='_single')
    
    all_paths = combPaths + combPaths_single
    print(all_paths)
    
    with tqdm(total=len(all_paths), desc="Processing files", unit="file") as pbar:
        for combPath, filename in all_paths:
            file_path = os.path.join(combPath, filename)
            process_and_check_data(file_path)
            pbar.update(1)  # Update the progress bar
