import numpy as np

# Load models
w_ols = np.load('w_ols.npy')
w_ridge = np.load('w_ridge_best.npy')
best_lambda = np.load('best_lambda.npy')

# Load test data
x_test, y_test = np.load("test.npz")["x"], np.load("test.npz")["y"]

# Create polynomial features
Phi_test = np.column_stack([x_test**i for i in range(10)])

# Predictions
y_pred_ols = Phi_test @ w_ols
y_pred_ridge = Phi_test @ w_ridge

# Compute MSE
mse_ols = np.mean((y_test - y_pred_ols) ** 2)
mse_ridge = np.mean((y_test - y_pred_ridge) ** 2)

# Report results
print("="*60)
print("PROBLEM 3.4: TEST SET EVALUATION")
print("="*60)
print(f"Test set size: {len(x_test)} samples")
print(f"\nOLS (λ=0):")
print(f"  Test MSE: {mse_ols:.6f}")
print(f"\nRidge (λ={best_lambda:.2e}):")
print(f"  Test MSE: {mse_ridge:.6f}")
print(f"\nDifference: {abs(mse_ols - mse_ridge):.6f}")
if mse_ridge < mse_ols:
    print(f"Ridge is better by {(mse_ols - mse_ridge) / mse_ols * 100:.2f}%")
else:
    print(f"OLS is better by {(mse_ridge - mse_ols) / mse_ridge * 100:.2f}%")
print("="*60)