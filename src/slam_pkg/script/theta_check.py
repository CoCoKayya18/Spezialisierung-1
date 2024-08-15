import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def process_and_check_data(file_path):
    try:
        data_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    
    columns_of_interest = ['Theta_calculated', 'yaw_world', 'angular_velocity_yaw', 'Time']
    
    if not all(col in data_df.columns for col in columns_of_interest):
        print(f"Columns of interest not found in {file_path}")
        return
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(file_path), "plots/theta_check")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Plot yaw_world vs theta_calculated to visually inspect alignment and jumps
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['yaw_world'].values, label='yaw_world')
    plt.plot(data_df['Theta_calculated'].values, label='theta_calculated')
    plt.xlabel('Index')
    plt.ylabel('Angle (radians)')
    plt.title('Yaw World vs Theta Calculated')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'yaw_vs_theta.png'))
    plt.close()

    # Plot the angular_velocity_yaw to check its relationship with theta jumps
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['angular_velocity_yaw'].values, label='angular_velocity_yaw')
    plt.xlabel('Index')
    plt.ylabel('Angular Velocity (radians/sec)')
    plt.title('Angular Velocity Yaw')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'angular_velocity_yaw.png'))
    plt.close()

    # Optional: Plot the error (difference) between theta_calculated and yaw_world
    data_df['error'] = data_df['yaw_world'] - data_df['Theta_calculated']
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['error'].values, label='error (yaw_world - theta_calculated)')
    plt.xlabel('Index')
    plt.ylabel('Error (radians)')
    plt.title('Error Between Yaw World and Theta Calculated')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'theta_error.png'))
    plt.close()

    print(f"Plots saved to {plots_dir}")

if __name__ == '__main__':
    all_paths = [
        ('../Spezialisierung-1/src/slam_pkg/data/random_single', 'FullData_single.csv'),
        ('../Spezialisierung-1/src/slam_pkg/data/random', 'FullData.csv'),
        ('../Spezialisierung-1/src/slam_pkg/data/random2_single', 'FullData_single.csv'),
        ('../Spezialisierung-1/src/slam_pkg/data/random2', 'FullData.csv'),
        ('../Spezialisierung-1/src/slam_pkg/data/random3_single', 'FullData_single.csv'),
        ('../Spezialisierung-1/src/slam_pkg/data/random3', 'FullData.csv')
    ]

    with tqdm(total=len(all_paths), desc="Processing files", unit="file") as pbar:
        for combPath, filename in all_paths:
            file_path = os.path.join(combPath, filename)
            process_and_check_data(file_path)
            pbar.update(1)  # Update the progress bar
