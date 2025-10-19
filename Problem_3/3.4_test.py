import numpy as np

# === Load test set ===
test = np.load("test.npz")
x_test, y_test = test["x"], test["y"]

# === Load trained models ===
models = np.load("ridge_models.npz")
w_ols_all = models["w_ols"]       # shape (5, 10)
w_ridge_all = models["w_ridge"]   # shape (5, 10)
best_lambda = models["best_lambda"]

# === Average across folds ===
w_ols = np.mean(w_ols_all, axis=0)
w_ridge = np.mean(w_ridge_all, axis=0)

# === Polynomial feature matrix ===
def build_features(x, deg=9):
    return np.array([[val**i for i in range(deg + 1)] for val in x])

# === MSE function ===
def calc_mse(preds, targets):
    return np.mean((preds - targets)**2)

# === Generate features and predict ===
X_test_poly = build_features(x_test)
y_pred_ols = X_test_poly @ w_ols
y_pred_ridge = X_test_poly @ w_ridge

# === Compute MSE ===
mse_ols = calc_mse(y_pred_ols, y_test)
mse_ridge = calc_mse(y_pred_ridge, y_test)

# === Display results ===
print(f"Test MSE - OLS (λ = 0): {mse_ols:.6f}")
print(f"Test MSE - Ridge (λ ≈ {best_lambda:.2e}): {mse_ridge:.6f}")
