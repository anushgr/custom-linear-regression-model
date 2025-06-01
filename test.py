import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def compute_cost(self, X, y, weights, bias):
        m = len(y)
        predictions = X.dot(weights) + bias
        errors = predictions - y
        cost = (1/(2*m)) * np.sum(errors**2)
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Forward pass: compute predictions
            predictions = X.dot(self.weights) + self.bias

            # Compute gradients
            dw = (1/m) * X.T.dot(predictions - y)
            db = (1/m) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Store cost for monitoring
            cost = self.compute_cost(X, y, self.weights, self.bias)
            self.cost_history.append(cost)

    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) * 0.5

# Reshape y to ensure it's 1D for plotting
y = y.ravel()

# Create and train the model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Save the plot
plt.show()
