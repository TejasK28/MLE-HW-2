import numpy as np
import matplotlib.pyplot as plt

data = np.load("train.npz")
x_train, y_train = data["x"], data["y"]


models = np.load("ridge_models.npz")
ols_weights = models["w_ols"]      # shape (5, 10)
ridge_weights = models["w_ridge"]  # shape (5, 10)
best_lambda = models["best_lambda"]

w_ols_mean = np.mean(ols_weights, axis=0)
w_ridge_mean = np.mean(ridge_weights, axis=0)

x_vals = np.linspace(0, 1, 200)
phi = np.array([[x**i for i in range(len(w_ols_mean))] for x in x_vals])

pred_ols = phi @ w_ols_mean
pred_ridge = phi @ w_ridge_mean
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color="gray", alpha=0.6, label="Training Points")
plt.plot(x_vals, pred_ols, color="blue", lw=2, label="OLS (λ = 0)")
plt.plot(x_vals, pred_ridge, color="red", lw=2,
         label=f"Ridge (λ ≈ {best_lambda:.2e})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("OLS vs Ridge Regression (Averaged over 5 folds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
