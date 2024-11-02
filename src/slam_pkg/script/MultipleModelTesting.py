import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import GPy
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

def plot_comparison(Y_true, Y_pred, target_names, metrics, plot_dir):
    for i, target_name in enumerate(target_names):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(Y_true[:, i], Y_pred[:, i], alpha=0.3, s=10)
        ax.plot([Y_true[:, i].min(), Y_true[:, i].max()], [Y_true[:, i].min(), Y_true[:, i].max()], 'k--', lw=2)
        ax.set_title(f'{target_name} Comparison\n'
                     f'MSE: {metrics[target_name]["MSE"]:.4f}, '
                     f'RMSE: {metrics[target_name]["RMSE"]:.4f}, '
                     f'MAE: {metrics[target_name]["MAE"]:.4f}, '
                     f'R2: {metrics[target_name]["R-squared"]:.4f}')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{target_name}_comparison_plot.png'))
        plt.close(fig)

def plot_comparison_for_paper(Y_true, Y_pred, target_names, metrics, plot_dir):
    for i, target_name in enumerate(target_names):

        if target_name == "delta_position_x_world":
            plot_name = "Δx"
        elif target_name == "delta_position_y_world":
            plot_name = "Δy"
        elif target_name == "delta_yaw":
            plot_name = "Δθ"

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(Y_true[:, i], Y_pred[:, i], alpha=0.3, s=10)
        ax.plot([Y_true[:, i].min(), Y_true[:, i].max()], [Y_true[:, i].min(), Y_true[:, i].max()], 'k--', lw=2)
        
        # Add grid and set text size
        ax.grid(True)
        ax.set_title(f'{plot_name} Comparison\n'
                     f'RMSE: {metrics[target_name]["RMSE"]:.4f}, '
                     f'R^2: {metrics[target_name]["R-squared"]:.4f}', fontsize=18)
        ax.set_xlabel('Ground Truth [m]', fontsize=16)
        ax.set_ylabel('Predicted Values [m]', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{plot_name}_comparison_plot.png'))
        plt.close(fig)

def plot_path_comparison(Y_true, Y_pred, plot_dir):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(np.cumsum(Y_true[:, 0]), np.cumsum(Y_true[:, 1]), label='True Path', color='blue')
    ax.plot(np.cumsum(Y_pred[:, 0]), np.cumsum(Y_pred[:, 1]), label='Predicted Path', color='red', linestyle='dashed')
    ax.set_title('True vs Predicted Path')
    ax.set_xlabel('Cumulative delta_position_x_world')
    ax.set_ylabel('Cumulative delta_position_y_world')
    ax.legend()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'path_comparison_plot.png'))
    plt.close()

def plot_residuals(Y_true, Y_pred, target_names, plot_dir):
    residuals = Y_true - Y_pred
    for i, target_name in enumerate(target_names):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(Y_pred[:, i], residuals[:, i], alpha=0.5)
        ax.hlines(y=0, xmin=Y_pred[:, i].min(), xmax=Y_pred[:, i].max(), color='red')
        ax.set_title(f'{target_name} Residuals')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{target_name}_residuals_plot.png'))
        plt.close(fig)

def plot_high_error_points(Y_true, Y_pred, X_test, features, target_names, plot_dir, threshold=0.01):
    residuals = Y_true - Y_pred
    for i, target_name in enumerate(target_names):
        high_error_indices = np.where(np.abs(residuals[:, i]) > threshold)[0]
        high_error_points = X_test[high_error_indices, :]
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(Y_true[:, i], Y_pred[:, i], alpha=0.3, s=10, label='Data Points')
        ax.scatter(Y_true[high_error_indices, i], Y_pred[high_error_indices, i], color='red', s=50, label='High Error Points')
        ax.plot([Y_true[:, i].min(), Y_true[:, i].max()], [Y_true[:, i].min(), Y_true[:, i].max()], 'k--', lw=2)
        ax.set_title(f'{target_name} High Error Points\n'
                     f'Threshold: {threshold}')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.legend()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{target_name}_high_error_points.png'))
        plt.close(fig)

def plot_error_distribution(Y_true, Y_pred, target_names, plot_dir):
    residuals = Y_true - Y_pred
    for i, target_name in enumerate(target_names):
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.kdeplot(residuals[:, i], ax=ax)
        ax.set_title(f'Error Distribution for {target_name}')
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{target_name}_error_distribution.png'))
        plt.close(fig)

def plot_histogram_residuals(Y_true, Y_pred, target_names, plot_dir):
    residuals = Y_true - Y_pred
    for i, target_name in enumerate(target_names):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.hist(residuals[:, i], bins=50, alpha=0.75, color='blue', edgecolor='black')
        ax.set_title(f'Histogram of {target_name} Residuals')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{target_name}_histogram_residuals.png'))
        plt.close()

def plot_index_comparison(Y_true, Y_pred, target_names, plot_dir, subsample_rate=0.1):
    indices = np.arange(len(Y_true))
    subsample_indices = indices[::int(1/subsample_rate)]  # Subsample the indices
    for i, target_name in enumerate(target_names):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(subsample_indices, Y_true[subsample_indices, i], label='True Values', color='blue', alpha=0.6)
        ax.plot(subsample_indices, Y_pred[subsample_indices, i], label='Predicted Values', color='red', linestyle='dashed', alpha=0.6)
        for j in subsample_indices:
            ax.vlines(j, min(Y_true[j, i], Y_pred[j, i]), max(Y_true[j, i], Y_pred[j, i]), color='gray', alpha=0.3)
        ax.set_title(f'{target_name} Index Comparison')
        ax.set_xlabel('Index')
        ax.set_ylabel('Values')
        ax.legend()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{target_name}_index_comparison_plot.png'))
        plt.close(fig)

def test_model_performance(datafilepath, modelFilePath, scalerFilePath, combPath, features, target, kinematic_deltas, model_type, single=False, data_type='calcJoint'):
    suffix = '_single' if single else ''
    direction = f'{combPath}{suffix}'

    if data_type == 'odomVel':
        odomSuffix = '_odometry'
    else:
        odomSuffix = ''

    # Load the test data
    test_data_path = os.path.join(datafilepath, direction, 'training', model_type, data_type, f'sparse_test_data_{combPath}{odomSuffix}{suffix}.csv')
    test_data = pd.read_csv(test_data_path)
    X_test = test_data[features].values
    Y_test = test_data[target].values

    # Load the scalers
    scaler_filenameX = os.path.join(scalerFilePath, direction, model_type, data_type, f'sparse_scaler_X_{combPath}{odomSuffix}{suffix}.pkl')
    scaler_filenameY = os.path.join(scalerFilePath, direction, model_type, data_type, f'sparse_scaler_Y_{combPath}{odomSuffix}{suffix}.pkl')

    with open(scaler_filenameX, 'rb') as file:
        scaler_X = pickle.load(file)
    with open(scaler_filenameY, 'rb') as file:
        scaler_Y = pickle.load(file)

    # Scale the test data
    X_test_scaled = scaler_X.transform(X_test)
    Y_test_scaled = scaler_Y.transform(Y_test)

    # Load the model
    model_filename = os.path.join(modelFilePath, direction, model_type, data_type, f'sparse_gpy_model_{combPath}{odomSuffix}{suffix}.pkl')
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    # Predict using the model
    Y_pred_scaled, _ = model.predict(X_test_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    # Calculate metrics for each target variable
    metrics = {}
    for i, target_name in enumerate(target):
        mse = mean_squared_error(Y_test[:, i], Y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test[:, i], Y_pred[:, i])
        metrics[target_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R-squared': r2
        }

    # Prepare the report content
    report_content = [
        f"Number of test data points: {len(Y_test)}"
    ]
    for target_name, target_metrics in metrics.items():
        report_content.append(f"{target_name} Metrics:")
        report_content.append(f"  Mean Squared Error (MSE): {target_metrics['MSE']}")
        report_content.append(f"  Root Mean Squared Error (RMSE): {target_metrics['RMSE']}")
        report_content.append(f"  Mean Absolute Error (MAE): {target_metrics['MAE']}")
        report_content.append(f"  R-squared: {target_metrics['R-squared']}")
        report_content.append("")

    # Print the report
    for line in report_content:
        print(line)

    # Save the report
    testing_dir = os.path.join(modelFilePath, direction, model_type, data_type, 'Testing')
    os.makedirs(testing_dir, exist_ok=True)
    report_filename = os.path.join(testing_dir, 'TestingReport.txt')
    with open(report_filename, 'w') as report_file:
        for line in report_content:
            report_file.write(line + '\n')

    # Plot comparisons
    # plot_comparison_dir = os.path.join(testing_dir, 'Plots', 'comparison_plots')
    # plot_comparison(Y_test, Y_pred, target, metrics, plot_comparison_dir)

    # plot_path_comparison_dir = os.path.join(testing_dir, 'Plots', 'path_comparison_plot')
    # plot_path_comparison(Y_test, Y_pred, plot_path_comparison_dir)

    # plot_residuals_dir = os.path.join(testing_dir, 'Plots', 'residuals_plots')
    # plot_residuals(Y_test, Y_pred, target, plot_residuals_dir)

    # plot_histogram_residuals_dir = os.path.join(testing_dir, 'Plots', 'histogram_residuals')
    # plot_histogram_residuals(Y_test, Y_pred, target, plot_histogram_residuals_dir)

    # plot_index_comparison_dir = os.path.join(testing_dir, 'Plots', 'index_comparison_plots')
    # plot_index_comparison(Y_test, Y_pred, target, plot_index_comparison_dir)

    # plot_high_error_points_dir = os.path.join(testing_dir, 'Plots', 'plot_high_error_points')
    # plot_high_error_points(Y_test, Y_pred, X_test, features, target, plot_high_error_points_dir)

    # plot_error_distribution_dir = os.path.join(testing_dir, 'Plots', 'plot_error_distribution')
    # plot_error_distribution(Y_test, Y_pred, target, plot_error_distribution_dir)

    plot_comparison_for_paper_dir = os.path.join(testing_dir, 'Plots', 'comparison_plots_for_paper')
    plot_comparison_for_paper(Y_test, Y_pred, target, metrics, plot_comparison_for_paper_dir)

if __name__ == '__main__':
    datafilepath = '../Spezialisierung-1/src/slam_pkg/data'
    modelFilePath = '../Spezialisierung-1/src/slam_pkg/myMLmodel'
    scalerFilePath = '../Spezialisierung-1/src/slam_pkg/Scaler'

    combPaths = [
        'x_direction_positive',
        'x_direction_negative',
        'y_direction_positive',
        'y_direction_negative',
        'diagonal_first_quad',
        'diagonal_second_quad',
        'diagonal_third_quad',
        'diagonal_fourth_quad',
        'diagonal_first_and_third_quad',
        'diagonal_second_and_fourth_quad',
        'square',
        'AllCombined',
        'random',
        'random2',
        'random3'
    ]

    features = ['world_velocity_x' , 'world_velocity_y', 'angular_velocity_yaw']
    target = ['delta_position_x_world', 'delta_position_y_world', 'delta_yaw']
    kinematic_deltas = ['kinematic_delta_x', 'kinematic_delta_y', 'kinematic_delta_yaw']

    # Define odometry features for the odometry models
    odometry_features = ['odom_world_velocity_x', 'odom_world_velocity_y', 'odom_angular_velocity']

    for combPath in tqdm(combPaths, desc="Testing models", unit="path"):
        # Test the standard model
        print(f"\nTesting model for {combPath} on FullData...")
        test_model_performance(datafilepath, modelFilePath, scalerFilePath, combPath, features, target, kinematic_deltas, 'FullData', data_type='calcJoint')

        print(f"\nTesting model for {combPath} on FullData_cleaned...")
        test_model_performance(datafilepath, modelFilePath, scalerFilePath, combPath, features, target, kinematic_deltas, 'CleanedData', data_type='calcJoint')

        print(f"\nTesting model for {combPath} on FullData_single...")
        test_model_performance(datafilepath, modelFilePath, scalerFilePath, combPath, features, target, kinematic_deltas, 'FullData', single=True, data_type='calcJoint')

        print(f"\nTesting model for {combPath} on FullData_cleaned_single...")
        test_model_performance(datafilepath, modelFilePath, scalerFilePath, combPath, features, target, kinematic_deltas, 'CleanedData', single=True, data_type='calcJoint')

        # Test the odometry model
        print(f"\nTesting odometry model for {combPath} on FullData...")
        test_model_performance(datafilepath, modelFilePath, scalerFilePath, combPath, odometry_features, target, kinematic_deltas, 'FullData', data_type='odomVel')

        print(f"\nTesting odometry model for {combPath} on FullData_cleaned...")
        test_model_performance(datafilepath, modelFilePath, scalerFilePath, combPath, odometry_features, target, kinematic_deltas, 'CleanedData', data_type='odomVel')

        print(f"\nTesting odometry model for {combPath} on FullData_single...")
        test_model_performance(datafilepath, modelFilePath, scalerFilePath, combPath, odometry_features, target, kinematic_deltas, 'FullData', single=True, data_type='odomVel')

        print(f"\nTesting odometry model for {combPath} on FullData_cleaned_single...")
        test_model_performance(datafilepath, modelFilePath, scalerFilePath, combPath, odometry_features, target, kinematic_deltas, 'CleanedData', single=True, data_type='odomVel')
