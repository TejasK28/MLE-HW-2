import numpy as np
import matplotlib.pyplot as plt

train = np.load("train.npz")
x_train, y_train = train["x"], train["y"]

test = np.load("test.npz")
x_test, y_test = test["x"], test["y"]

def make_poly(x, deg=9):
    return np.array([[xi**d for d in range(deg + 1)] for xi in x])

def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.pinv(X.T @ X + lam * np.eye(d)) @ X.T @ y

def calc_mse(pred, actual):
    return np.mean((pred - actual) ** 2)

def predict(X, w):
    return X @ w

def kfold_indices(n, k=5, seed=123):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    chunks = np.array_split(idx, k)
    for i in range(k):
        val = chunks[i]
        train = np.hstack(chunks[:i] + chunks[i+1:])
        yield train, val

def ridge_cv(x, y, lambdas, k=5):
    val_mses, model_grid = [], []

    for lam in lambdas:
        fold_mses, fold_ws = [], []
        for tr_idx, val_idx in kfold_indices(len(x), k):
            x_tr, y_tr = x[tr_idx], y[tr_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            phi_tr = make_poly(x_tr)
            phi_val = make_poly(x_val)

            w = ridge_fit(phi_tr, y_tr, lam)
            fold_ws.append(w)
            fold_mses.append(calc_mse(predict(phi_val, w), y_val))

        val_mses.append(np.mean(fold_mses))
        model_grid.append(np.array(fold_ws))

    return np.array(val_mses), model_grid

lambdas = np.concatenate([np.logspace(-8, -1, 30), np.linspace(0.1, 1, 20)])
val_mses, models = ridge_cv(x_train, y_train, lambdas)

best_idx = np.argmin(val_mses)
best_lambda = lambdas[best_idx]
print(f"\nBest λ = {best_lambda:.2e}")
print(f"Validation MSE = {val_mses[best_idx]:.5f}")

ridge_models = models[best_idx]
_, ols_models_list = ridge_cv(x_train, y_train, np.array([0.0]))
ols_models = ols_models_list[0]

np.savez("ridge_models.npz", w_ols=ols_models, w_ridge=ridge_models, best_lambda=best_lambda)

plt.figure(figsize=(8, 5))
plt.plot(lambdas, val_mses, "o-", label="Validation MSE")
plt.axvline(best_lambda, color="green", ls="--", label=f"Best λ = {best_lambda:.2e}")
plt.xscale("log")
plt.xlabel("λ (Regularization strength)")
plt.ylabel("MSE")
plt.title("Ridge Regression (5-Fold CV)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
