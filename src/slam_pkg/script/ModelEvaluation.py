import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


ith_datapoint = 1
isSparse = 'sparseKFoldDiagonalFirstDirection_'
# isSparse = ''
SpecialCase = '_First_Diagonal_Direction'
# SpecialCase = ''
# dataName = 'Data.csv'
dataName = 'Data_Square_Direction'
# isTuned = 'BayesianOptimizationTuned_'
# isTuned = 'GridSearchTuned_'
# isTuned = 'BayesianOptimizationTuned_GridSearchTuned_'
isTuned = ''
# trainOrTest = '_train'
trainOrTest = '_test'

# /home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/sparseKFoldDiagonal_1_Diagonal_Direction_DP_predictions_vs_real_test.csv

# Load the standardized predictions and real values
filepath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isTuned}{isSparse}{ith_datapoint}{SpecialCase}_DP_predictions_vs_real{trainOrTest}.csv'

data = pd.read_csv(filepath)

if data.isna().any().any():
    print("NaN values found in the DataFrame. They will be dropped for model training.")
    data = data.dropna()

features = ['X', 'Y', 'Yaw']

# Lists to hold figure titles and error metrics text
figure_titles = []
error_texts = []

r2_scores = []
mse_scores = []
mae_scores = []
rmse_scores = []

# Iterate over each feature to fit a model, make predictions, and calculate metrics
for i, feat in enumerate(features, start=1):
    # Predictor and response variables
    X = data[[f'Real_{feat}']].values.reshape(-1, 1)
    y = data[f'Predicted_{feat}']

    # Fit the linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    data[f'Predicted_{feat}_LM'] = y_pred

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rms = rmse(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)

    # Store the figure title and error metrics text
    figure_titles.append(f'GT Δ vs Predicted Δ {feat}')
    error_texts.append(f'R²: {r2:.2f}\nRMSE: {rms:.4f}')

# Plotting
plt.figure(figsize=(15, 10))

for i, feat in enumerate(features, start=1):
    plt.subplot(len(features), 1, i)
    plt.scatter(data[f'Real_{feat}'], data[f'Predicted_{feat}'], alpha=0.1)  # Plot real vs predicted with transparency
    plt.plot(data[f'Real_{feat}'].values, data[f'Predicted_{feat}_LM'].values, 'r--')  # Plot linear model predictions
    # plt.title(figure_titles[i-1])
    plt.xlabel(f'GT Δ{feat} [m]')
    plt.ylabel(f'Predicted Δ{feat} [m]')
    plt.text(0.05, 0.95, error_texts[i-1], transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.grid(True)

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.30, top=0.70, wspace=0.20, hspace=0.22)
plt.tight_layout()
plt.show()


# for i, feats in enumerate(features, start=1):
#     # Fit a linear model
#     X = data[[f'Real_{feats}']]  # Predictor variable
#     y = data[f'Predicted_{feats}']  # Response variable
#     model = LinearRegression().fit(X, y)

#     # Predictions using the linear model
#     data[f'Predicted_{feats}_LM'] = model.predict(X)

#     # Calculate the R^2 value
#     r2 = r2_score(y, data[f'Predicted_{feats}_LM'])
#     print(f'R^2 value: {r2}')

#     # Plotting real X vs. predicted X
#     plt.subplot(len(features), 1, i)  # 3 rows, 1 column, ith subplot
#     plt.scatter(data[f'Real_{feats}'], data[f'Predicted_{feats}'], alpha=0.1)  # Plot with some transparency for overlapping points
#     plt.plot(data[f'Real_{feats}'].values, data[f'Predicted_{feats}_LM'].values, 'r--')  # Add the linear model predictions as a red dashed line
#     plt.title(f'Real {feats} vs. Predicted {feats}')  # Title of the subplot
#     plt.xlabel(f'Real {feats}')  # X-axis label
#     plt.ylabel(f'Predicted {feats}')  # Y-axis label
#     plt.grid(True)  # Add a grid for easier visualization
#     plt.text(0.05, 0.95, f'R² value: {r2:.2f}', transform=plt.gca().transAxes)

# plt.show()