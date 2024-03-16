import numpy as np
import matplotlib.pyplot as plt
import GPy
import pickle
import os

ith_datapoint = 100

model_filename = f'gpy_model_{ith_datapoint}DP.pkl'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'

with open(os.path.join(modelFilePath, model_filename), 'rb') as file:
    gp_model = pickle.load(file)


# Assuming you have a GP model called 'gp_model' and it's already trained
# Generate some test points (input values) for predictions
X_test = np.linspace(start, end, num_points)[:, None]  # Reshape if just one feature for GPy

# Make predictions
mean_prediction, variance_prediction = gp_model.predict(X_test)

# Calculate the 95% confidence intervals for the predictions
confidence_upper = mean_prediction + 1.96 * np.sqrt(variance_prediction)
confidence_lower = mean_prediction - 1.96 * np.sqrt(variance_prediction)

# Plot the mean prediction
plt.plot(X_test, mean_prediction, 'b-', label='Mean Prediction')

# Plot the confidence interval
plt.fill_between(X_test.flatten(), confidence_lower.flatten(), confidence_upper.flatten(), alpha=0.3, label='95% Confidence Interval')

# If you have actual values to compare against, plot them too
plt.plot(X_actual, Y_actual, 'rx', label='Actual Values')

plt.xlabel('Input Feature')
plt.ylabel('Target Variable')
plt.title('Gaussian Process Regression Model Predictions')
plt.legend()

plt.show()
