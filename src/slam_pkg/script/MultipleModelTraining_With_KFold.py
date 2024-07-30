import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import GPy
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_feature_distribution(X_train, X_test, feature_names, title_suffix='', plot_dir=None):
    num_features = len(feature_names)
    fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 4))
    
    if num_features == 1:
        axes = [axes]
    
    for i, feature_name in enumerate(feature_names):
        axes[i].hist(X_train[:, i], bins=20, color='blue', alpha=0.5, label='Train')
        axes[i].hist(X_test[:, i], bins=20, color='orange', alpha=0.5, label='Test')
        axes[i].set_title(f'{feature_name} Distribution {title_suffix}')
        axes[i].set_xlabel(feature_name)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    plt.tight_layout()
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'feature_distribution{title_suffix}.png'))
    else:
        plt.show()
    plt.close(fig)

def check_standardization(data, tolerance=1e-6):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return np.all(np.abs(mean) < tolerance) and np.all(np.abs(std - 1.0) < tolerance)

def train_and_evaluate_model(dataframe, features, target, kinematic_deltas, SpecialCase='', direction=''):
    X = dataframe[features].values
    Y = dataframe[target].values
    kinematic_data = dataframe[kinematic_deltas].values
    
    X_train_full, X_test, Y_train_full, Y_test, kinematic_train_full, kinematic_test = train_test_split(X, Y, kinematic_data, test_size=0.3, random_state=42)

    # Plotting target distributions
    plot_dir = os.path.join(datafilepath, direction, 'plots', 'Training')
    plot_feature_distribution(X_train_full, X_test, features, title_suffix='(Features)', plot_dir=plot_dir)
    plot_feature_distribution(Y_train_full, Y_test, target, title_suffix='(Targets)', plot_dir=plot_dir)

    # Then fit and transform with standardization
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_full = scaler_X.fit_transform(X_train_full)
    X_test = scaler_X.transform(X_test)  # Only transform the test data
    Y_train_full = scaler_Y.fit_transform(Y_train_full)
    Y_test = scaler_Y.transform(Y_test)  # Only transform the test data

    # Check if data is standardized correctly
    if not (check_standardization(X_train_full) and check_standardization(Y_train_full)):
        print(f"Standardization check failed for {direction}. Skipping model training.")
        return

    # Number of folds
    k = 5  

    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    best_r2 = float('-inf')  # Initialize to negative infinity
    best_model = None
    best_indices = None

    # Lists to store results of each fold
    fold_mse = []
    fold_rmse = []
    fold_mae = []
    fold_r2 = []

    # Iterate over each split
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(X_train_full), total=k, desc="Folds")):
        print(f"\n Training on fold {i+1}/{k}...")  # Print the current fold number

        # Split the data
        X_train, X_val = X_train_full[train_index], X_train_full[val_index]
        Y_train, Y_val = Y_train_full[train_index], Y_train_full[val_index]

        # Define the kernel for GPy model
        kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=1.0, lengthscale=1.0) + GPy.kern.White(X_train.shape[1], variance=1.0)

        # Create the sparse model
        model = GPy.models.SparseGPRegression(X_train, Y_train, kernel)

        # Optimize Hyperparameters
        model.optimize()
        model.optimize_restarts(num_restarts=10, verbose=False)

        # Evaluate the model on the validation set
        preds, _ = model.predict(X_val)

        mse = mean_squared_error(Y_val, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_val, preds)
        r2 = r2_score(Y_val, preds)

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_indices = (train_index, val_index)
        
        fold_mse.append(mse)
        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_r2.append(r2)

    # Calculate average performance across all folds
    average_mse = np.mean(fold_mse)
    average_rmse = np.mean(fold_rmse)
    average_mae = np.mean(fold_mae)
    average_r2 = np.mean(fold_r2)

    print("Average MSE across all folds:", average_mse)
    print("Average RMSE across all folds:", average_rmse)
    print("Average MAE across all folds:", average_mae)
    print("Average R-squared across all folds:", average_r2)

    print("MSE across all folds:", fold_mse)
    print("RMSE across all folds:", fold_rmse)
    print("MAE across all folds:", fold_mae)
    print("R-squared across all folds:", fold_r2)

    print(best_model)

    # Save the model and scalers
    model_dir = os.path.join(modelFilePath, direction)
    scaler_dir = os.path.join(scalerFilePath, direction)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)

    model_filename = f'sparse_gpy_model_{SpecialCase}.pkl'
    scaler_filenameX = f'sparse_scaler_X_{SpecialCase}.pkl'
    scaler_filenameY = f'sparse_scaler_Y_{SpecialCase}.pkl'

    with open(os.path.join(model_dir, model_filename), 'wb') as file:
        pickle.dump(best_model, file)

    with open(os.path.join(scaler_dir, scaler_filenameX), 'wb') as file:
        pickle.dump(scaler_X, file)

    with open(os.path.join(scaler_dir, scaler_filenameY), 'wb') as file:
        pickle.dump(scaler_Y, file)

    # Prepare datasets for the best model
    X_train, X_val = X_train_full[best_indices[0]], X_train_full[best_indices[1]]
    Y_train, Y_val = Y_train_full[best_indices[0]], Y_train_full[best_indices[1]]

    # Inverse transform features and targets for saving to CSV
    X_train_inv = scaler_X.inverse_transform(X_train)
    X_val_inv = scaler_X.inverse_transform(X_val)
    X_test_inv = scaler_X.inverse_transform(X_test)

    Y_train_inv = scaler_Y.inverse_transform(Y_train)
    Y_val_inv = scaler_Y.inverse_transform(Y_val)
    Y_test_inv = scaler_Y.inverse_transform(Y_test)

    # Save data for the best model
    # Concatenate the inversely transformed data and kinematic data
    train_data = pd.concat([pd.DataFrame(X_train_inv, columns=features), pd.DataFrame(Y_train_inv, columns=target), pd.DataFrame(kinematic_train_full[best_indices[0]], columns=kinematic_deltas)], axis=1)

    val_data = pd.concat([pd.DataFrame(X_val_inv, columns=features), pd.DataFrame(Y_val_inv, columns=target), pd.DataFrame(kinematic_train_full[best_indices[1]], columns=kinematic_deltas)], axis=1)

    test_data = pd.concat([pd.DataFrame(X_test_inv, columns=features), pd.DataFrame(Y_test_inv, columns=target), pd.DataFrame(kinematic_test, columns=kinematic_deltas)], axis=1)

    train_data.to_csv(os.path.join(datafilepath, direction, f'sparse_train_data_{SpecialCase}.csv'), index=False)
    val_data.to_csv(os.path.join(datafilepath, direction, f'sparse_val_data_{SpecialCase}.csv'), index=False)
    test_data.to_csv(os.path.join(datafilepath, direction, f'sparse_test_data_{SpecialCase}.csv'), index=False)

if __name__ == '__main__':
    datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
    modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
    scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'

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
        'AllCombined'
    ]

    features = ['world_velocity_x' , 'world_velocity_y', 'angular_velocity_yaw']
    target = ['delta_position_x_world', 'delta_position_y_world', 'delta_yaw']
    kinematic_deltas = ['kinematic_delta_x', 'kinematic_delta_y', 'kinematic_delta_yaw']

    for combPath in tqdm(combPaths, desc="Training models", unit="path"):
        dataName = 'FullData.csv'
        dataframe = pd.read_csv(os.path.join(datafilepath, combPath, dataName))
        print(f"\nTraining model for {combPath}...")
        train_and_evaluate_model(dataframe, features, target, kinematic_deltas, SpecialCase=combPath, direction=combPath)
