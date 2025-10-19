import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

np.random.seed(69)

def load_npz(fname):
    data = np.load(fname, allow_pickle=True)
    return data["logit"], data["year"], data["filename"]

x_train, y_train, f_train = load_npz("vgg16_train.npz")
x_test, y_test, f_test = load_npz("vgg16_test.npz")

def pca_whiten(x, k):
    mu = x.mean(axis=0, keepdims=True)
    x_c = x - mu
    _, s, vt = np.linalg.svd(x_c, full_matrices=False)
    vecs = vt.T[:, :k]
    vals = (s**2) / (x.shape[0] - 1)
    z = (x_c @ vecs) / np.sqrt(vals[:k])
    return z, mu, vecs, vals[:k]

z1, *_ = pca_whiten(x_train, 1)
z2, *_ = pca_whiten(x_train, 2)
z2_test, *_ = pca_whiten(x_test, 2)

def show_pca(z, years, title, mode=2):
    cmap = plt.cm.plasma
    norm = colors.Normalize(vmin=1148, vmax=2012)

    if mode == 1:
        plt.figure(figsize=(7, 4))
        plt.scatter(z[:, 0], years, c=years, cmap=cmap, norm=norm, s=5, edgecolor='k', linewidth=0.2)
        plt.xlabel("PC1")
        plt.ylabel("Year")
        plt.title(title)
        plt.colorbar(label="Year")
        plt.tight_layout()
        plt.show()

    elif mode == 3:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(z[:, 0], z[:, 1], years, c=years, cmap=cmap, norm=norm, s=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("Year")
        fig.colorbar(sc, label="Year")
        plt.title(title)
        plt.tight_layout()
        plt.show()

show_pca(z1, y_train, "PCA (1D)", mode=1)
show_pca(z2, y_train, "PCA (2D)", mode=3)

def poly_feats(X, deg=3):
    x1, x2 = X[:, 0], X[:, 1]
    feats = [np.ones(len(X))]
    for d in range(1, deg + 1):
        for i in range(d + 1):
            feats.append(x1**(d - i) * x2**i)
    return np.column_stack(feats)

def fit_mmse(X, y, deg=2):
    lams = np.logspace(-8, 0, 20)
    idx = np.random.permutation(len(X))
    cut = int(0.8 * len(X))
    X_tr, X_val = X[idx[:cut]], X[idx[cut:]]
    y_tr, y_val = y[idx[:cut]], y[idx[cut:]]

    best_w, best_err, best_lam = None, float('inf'), None
    all_mses = []

    for lam in lams:
        Xp = poly_feats(X_tr, deg)
        A = Xp.T @ Xp + lam * np.eye(Xp.shape[1])
        w = np.linalg.pinv(A) @ Xp.T @ y_tr

        preds = poly_feats(X_val, deg) @ w
        mse = np.mean((preds - y_val)**2)
        all_mses.append(mse)

        if mse < best_err:
            best_err, best_lam, best_w = mse, lam, w

    return best_lam, best_err, best_w, all_mses

def sweep_degrees(Z_tr, Z_te, y_tr, y_te, max_deg=5):
    degs = range(1, max_deg + 1)
    val_errs, test_errs, all_ws = [], [], []

    for d in degs:
        lam, val_mse, w, _ = fit_mmse(Z_tr, y_tr, d)
        y_pred = poly_feats(Z_te, d) @ w
        test_mse = np.mean((y_pred - y_te)**2)

        print(f"Degree {d} | Lambda: {lam:.2e} | Val MSE: {val_mse:.2f} | Test MSE: {test_mse:.2f}")

        val_errs.append(val_mse)
        test_errs.append(test_mse)
        all_ws.append(w)

    return degs, val_errs, test_errs, all_ws

degrees, val_mse_list, test_mse_list, weights = sweep_degrees(z2, z2_test, y_train, y_test)

def plot_mse_curve(deg_list, val_mse, test_mse):
    plt.figure(figsize=(9, 5))
    plt.plot(deg_list, val_mse, label="Val MSE", marker="s", linestyle="--", color="navy")
    plt.plot(deg_list, test_mse, label="Test MSE", marker="o", linestyle="-", color="darkred")

    best = np.argmin(test_mse)
    plt.scatter(deg_list[best], test_mse[best], color="green", s=80,
                label=f"Best Test MSE: {test_mse[best]:.2f}")

    plt.xlabel("Poly Degree")
    plt.ylabel("MSE")
    plt.title("MSE vs Degree")
    plt.xticks(deg_list)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best

best_idx = plot_mse_curve(degrees, val_mse_list, test_mse_list)

print("Best Degree:", degrees[best_idx])
print("Test MSE at Best Degree:", test_mse_list[best_idx])
print("Validation MSE at Best Degree:", val_mse_list[best_idx])

def error_extremes(Z_te, y_te, names, d, w):
    feats = poly_feats(Z_te, d)
    preds = feats @ w
    errs = np.abs(preds - y_te)

    best = np.argmin(errs)
    worst = np.argmax(errs)

    print("Most Accurate Prediction Image:", names[best])
    print("Least Accurate Prediction Image:", names[worst])

error_extremes(z2_test, y_test, f_test, degrees[best_idx], weights[best_idx])
