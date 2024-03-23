import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

ith_datapoint = 10
# isSparse = 'sparse0_'
isSparse = ''
isTuned = 'BayesianOptimizationTuned_'
# isTuned = 'GridSearchTuned_'
# isTuned = 'BayesianOptimizationTuned_GridSearchTuned_'
isTuned = ''

# Load the standardized predictions and real values
filepath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/TestingData/{isTuned}{isSparse}{ith_datapoint}_DP_predictions_vs_real_test.csv'
data = pd.read_csv(filepath)

features = ['X', 'Y', 'Yaw']

# # Assuming 'data' is your DataFrame and contains features you want to scale
# features_to_scale = [f'Real_{feat}' for feat in features] + [f'Predicted_{feat}' for feat in features]

# # Initialize the scaler with your desired range
# scaler = MinMaxScaler(feature_range=(-0.01, 0.01))

# # Fit the scaler to your data and transform it
# data_scaled = scaler.fit_transform(data[features_to_scale])

# # The output is a numpy array, so if you want to update your original DataFrame:
# data[features_to_scale] = data_scaled

for i, feats in enumerate(features, start=1):
    # Fit a linear model
    X = data[[f'Real_{feats}']]  # Predictor variable
    y = data[f'Predicted_{feats}']  # Response variable
    model = LinearRegression().fit(X, y)

    # Predictions using the linear model
    data[f'Predicted_{feats}_LM'] = model.predict(X)

    # Calculate the R^2 value
    r2 = r2_score(y, data[f'Predicted_{feats}_LM'])
    print(f'R^2 value: {r2}')

    # Plotting real X vs. predicted X
    plt.subplot(len(features), 1, i)  # 3 rows, 1 column, ith subplot
    plt.scatter(data[f'Real_{feats}'], data[f'Predicted_{feats}'], alpha=0.1)  # Plot with some transparency for overlapping points
    plt.plot(data[f'Real_{feats}'].values, data[f'Predicted_{feats}_LM'].values, 'r--')  # Add the linear model predictions as a red dashed line
    plt.title(f'Real {feats} vs. Predicted {feats}')  # Title of the subplot
    plt.xlabel(f'Real {feats}')  # X-axis label
    plt.ylabel(f'Predicted {feats}')  # Y-axis label
    plt.grid(True)  # Add a grid for easier visualization
    plt.text(0.05, 0.95, f'R² value: {r2:.2f}', transform=plt.gca().transAxes)

plt.show()