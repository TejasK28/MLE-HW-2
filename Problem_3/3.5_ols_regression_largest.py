# ols_regression_largeset.py - Problem 3.5: OLS on large dataset

import numpy as np
import matplotlib.pyplot as plt

# Load large training dataset
x_train, y_train = np.load("train_100.npz")["x"], np.load("train_100.npz")["y"]

print(f"Training on large dataset: {len(x_train)} samples")

# Create polynomial features (degree 9)
Phi_train = np.column_stack([x_train**i for i in range(10)])

# Train OLS (no regularization, no cross-validation)
w = np.linalg.solve(Phi_train.T @ Phi_train, Phi_train.T @ y_train)

# Compute training MSE
y_pred_train = Phi_train @ w
train_mse = np.mean((y_train - y_pred_train) ** 2)
print(f"Training MSE: {train_mse:.6f}")

# Generate predictions over [0, 1] for plotting
x_plot = np.linspace(0, 1, 1000)
Phi_plot = np.column_stack([x_plot**i for i in range(10)])
y_plot = Phi_plot @ w

# Plot
plt.figure(figsize=(14, 8))
plt.scatter(x_train, y_train, alpha=0.3, s=20, color='blue', 
           label=f'Training Data (N={len(x_train)})', edgecolors='none')
plt.plot(x_plot, y_plot, 'r-', linewidth=3, alpha=0.9, 
        label='OLS Model (degree 9, λ=0)')
plt.xlabel('x', fontsize=14, fontweight='bold')
plt.ylabel('y', fontsize=14, fontweight='bold')
plt.title('OLS Regression on Large Dataset (No Regularization)', 
         fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('ols_largeset.png', dpi=300, bbox_inches='tight')
print("Saved: ols_largeset.png")
plt.show()

# Save model
np.save('w_largeset.npy', w)
print("Saved: w_largeset.npy")