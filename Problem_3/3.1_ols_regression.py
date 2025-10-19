import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Loading datasets...")
train_data_small = np.load("train.npz")
train_x = train_data_small["x"]
train_y = train_data_small["y"]

test_data = np.load("test.npz")
test_x = test_data["x"]
test_y = test_data["y"]

train_data_large = np.load("train_100.npz")
train_x_100 = train_data_large["x"]
train_y_100 = train_data_large["y"]

def create_polynomial_features(X, degree=9):
    return np.array([[x**i for i in range(degree + 1)] for x in X])

def compute_ols_weights(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def make_predictions(X, weights):
    return X @ weights

def calculate_mse(pred, true):
    return np.mean((pred - true)**2)

def generate_kfold_splits(n, k=5, seed=123):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack(folds[:i] + folds[i+1:])
        splits.append((train_idx, val_idx))
    return splits

def perform_ols_cv(X, y, splits):
    mse_scores = []
    weights_list = []
    
    for train_idx, val_idx in splits:
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        train_features = create_polynomial_features(X_train)
        weights = compute_ols_weights(train_features, y_train)
        
        val_features = create_polynomial_features(X_val)
        predictions = make_predictions(val_features, weights)
        
        mse_scores.append(calculate_mse(predictions, y_val))
        weights_list.append(weights)
    
    return np.mean(mse_scores), np.mean(weights_list, axis=0)

cv_splits = generate_kfold_splits(len(train_x))
ols_avg_mse, ols_weights = perform_ols_cv(train_x, train_y, cv_splits)

print(f"OLS Average MSE: {ols_avg_mse:.6f}")
print(f"Features: {len(ols_weights)}")