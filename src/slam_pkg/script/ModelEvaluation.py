import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

ith_datapoint = 1
isSparse = 'sparseKFoldSquareDirection_'
# isSparse = ''
SpecialCase = '_Square_Direction'
# SpecialCase = ''
# dataName = 'Data.csv'
dataName = 'Data_Square_Direction_Direction.csv'
# isTuned = 'BayesianOptimizationTuned_'
# isTuned = 'GridSearchTuned_'
# isTuned = 'BayesianOptimizationTuned_GridSearchTuned_'
isTuned = ''
# trainOrTest = '_train'
trainOrTest = '_test'

# Load the standardized predictions and real values
filepath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isTuned}{isSparse}{ith_datapoint}{SpecialCase}_DP_predictions_vs_real{trainOrTest}.csv'

data = pd.read_csv(filepath)

if data.isna().any().any():
    print("NaN values found in the DataFrame. They will be dropped for model training.")
    data = data.dropna()

features = ['X', 'Y', 'Yaw']

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