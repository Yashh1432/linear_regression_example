import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate realistic sample data: House size (sq ft) vs. Price (USD)
np.random.seed(42)
X = np.linspace(800, 4000, 100).reshape(-1, 1)  # House sizes from 800 to 4000 sq ft
y = 100 * X + 50000 + np.random.normal(0, 30000, (100, 1))  # Price = 100 * size + 50k + noise

# Create and train the model
model = LinearRegression()
model.fit(X, y.ravel())

# Make predictions
y_pred = model.predict(X)

# Get the slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual house prices')
plt.plot(X, y_pred, color='red', label=f'Regression line: Price = {slope:.2f} * Size + {intercept:.2f}')
plt.title('House Price vs. Size (Simple Linear Regression)')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig('house_price_regression.png')
plt.show()

# Print model details
print(f"Slope (Price per sq ft): ${slope:.2f}")
print(f"Intercept (Base price): ${intercept:.2f}")