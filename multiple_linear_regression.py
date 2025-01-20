import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 samples, 3 features
true_coefficients = [5, 2, -1]  # True coefficients for the features
y = 5 + X.dot(true_coefficients) + np.random.randn(100) * 0.5  # Add some noise

# Add bias term (intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones to X for the intercept

# Calculate coefficients using the normal equation
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Print results
print("Intercept and Coefficients:", theta)

# Predictions
y_pred = X_b.dot(theta) 

# Plot true vs predicted
plt.scatter(y, y_pred, color='blue', alpha=0.7, label='True vs Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', label='Perfect Fit')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.title('True vs Predicted Values (Manual)')
plt.show()
