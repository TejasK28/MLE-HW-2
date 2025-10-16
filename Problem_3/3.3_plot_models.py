# plot_models.py - Problem 3.3 (Concise version)

import numpy as np
import matplotlib.pyplot as plt

# Load models and data
w_ols = np.load('w_ols.npy')
w_ridge = np.load('w_ridge_best.npy')
best_lambda = np.load('best_lambda.npy')
x_train, y_train = np.load("train.npz")["x"], np.load("train.npz")["y"]

# Generate predictions over [0, 1]
x_plot = np.linspace(0, 1, 1000)
Phi = np.column_stack([x_plot**i for i in range(10)])

y_ols = Phi @ w_ols
y_ridge = Phi @ w_ridge
y_avg = Phi @ np.mean([w_ols, w_ridge], axis=0)

# Plot
plt.figure(figsize=(14, 8))
plt.scatter(x_train, y_train, alpha=0.6, s=80, color='gray', 
           label='Training Data', zorder=5, edgecolors='black', linewidth=0.5)
plt.plot(x_plot, y_ols, 'b-', linewidth=3, alpha=0.8, label='OLS (λ=0)')
plt.plot(x_plot, y_ridge, 'r-', linewidth=3, alpha=0.8, label=f'Ridge (λ={best_lambda:.2e})')
plt.plot(x_plot, y_avg, 'g--', linewidth=3, alpha=0.8, label='Average')
plt.xlabel('x', fontsize=14, fontweight='bold')
plt.ylabel('y', fontsize=14, fontweight='bold')
plt.title('OLS vs Ridge Regression', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.show()