import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate some synthetic data for the demonstration
np.random.seed(42)  # For reproducibility
X = np.random.rand(100, 1) * 10  # 100 data points, single feature
y = 3.5 * X.squeeze() + np.random.randn(100) * 2  # Linear relation with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Example usage: predicting new data points
new_data = np.array([[5], [7]])
predictions = model.predict(new_data)
print("Predictions for new data:", predictions)