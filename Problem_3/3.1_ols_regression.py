import numpy as np

# TODO: Implement MMSE regression with 5-fold cross validation
# Using polynomial features up to degree 9
# Need to report average validation error across all folds

# Load training data
data = np.load("train.npz")
x_data = data["x"]  # input features
y_data = data["y"]  # target outputs (looks like noisy sine wave from the comment)

# Setup for cross validation
num_samples = len(x_data)
num_folds = 5
samples_per_fold = num_samples // num_folds  # might leave some samples out if not divisible
poly_degree = 9  # using 9th degree polynomial

# Randomize the data order
np.random.seed(42)  # for reproducibility
shuffled_indices = np.random.permutation(num_samples)

# Store MSE for each fold
validation_errors = []

print("Starting 5-fold cross validation...")

for current_fold in range(num_folds):
    print(f"Processing fold {current_fold + 1}/{num_folds}")
    
    # Figure out validation set indices for this fold
    start_idx = current_fold * samples_per_fold
    if current_fold == num_folds - 1:  # last fold gets any remaining samples
        end_idx = num_samples
    else:
        end_idx = (current_fold + 1) * samples_per_fold
    
    validation_indices = shuffled_indices[start_idx:end_idx]
    
    # Training set is everything else
    train_indices = np.concatenate([
        shuffled_indices[:start_idx], 
        shuffled_indices[end_idx:]
    ])
    
    # Extract actual data points
    x_train = x_data[train_indices]
    y_train = y_data[train_indices]
    x_val = x_data[validation_indices]
    y_val = y_data[validation_indices]
    
    # Build polynomial feature matrix: [1, x, x^2, x^3, ..., x^9]
    # For training data
    train_features = []
    for power in range(poly_degree + 1):
        train_features.append(x_train ** power)
    Phi_train = np.column_stack(train_features)
    
    # For validation data
    val_features = []
    for power in range(poly_degree + 1):
        val_features.append(x_val ** power)
    Phi_val = np.column_stack(val_features)
    
    # Solve normal equations: w = (Phi^T * Phi)^(-1) * Phi^T * y
    # This is the MMSE solution
    gram_matrix = Phi_train.T @ Phi_train
    rhs = Phi_train.T @ y_train
    weights = np.linalg.solve(gram_matrix, rhs)
    
    # Make predictions on validation set
    predictions = Phi_val @ weights
    
    # Calculate mean squared error
    squared_errors = (y_val - predictions) ** 2
    fold_mse = np.mean(squared_errors)
    validation_errors.append(fold_mse)
    
    print(f"Fold {current_fold + 1} MSE: {fold_mse:.6f}")

# Calculate final result
average_validation_mse = np.mean(validation_errors)
print(f"\n=== RESULTS ===")
print(f"Average validation MSE across {num_folds} folds: {average_validation_mse:.6f}")

# Might be useful to see the variance too
mse_std = np.std(validation_errors)
print(f"Standard deviation of MSE: {mse_std:.6f}")