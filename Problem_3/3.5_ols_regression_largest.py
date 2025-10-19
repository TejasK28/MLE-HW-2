import numpy as np
import matplotlib.pyplot as plt

# Load data
d = np.load("train_100.npz")
x, y = d["x"], d["y"]

# Build polynomial feature matrix (degree 9)
def make_features(x_vals, deg=9):
    return np.array([[x**i for i in range(deg + 1)] for x in x_vals])

# Solve for weights using basic OLS
def fit_ols(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y  # works well enough for now

# Train model
X_poly = make_features(x)
w = fit_ols(X_poly, y)

# Generate smooth prediction curve
x_smooth = np.linspace(0, 1, 200)
X_smooth = make_features(x_smooth)
y_smooth = X_smooth @ w

# Plot it
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="gray", alpha=0.5, label="Training")
plt.plot(x_smooth, y_smooth, color="blue", lw=2, label="OLS Fit")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("OLS on Larger Dataset (No Reg)")
plt.legend()
plt.grid(True)
plt.show()
