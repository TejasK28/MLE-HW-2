# ridge_regression.py - Problem 3.2 
# Ridge regression with cross-validation to find optimal lambda
# Author: Me (working on ML assignment)

import numpy as np
import matplotlib.pyplot as plt

# Load the training data
print("Loading training data...")
train_data = np.load("train.npz")
x_values = train_data["x"]
y_values = train_data["y"]

# Setup parameters
total_samples = len(x_values)
cv_folds = 5  # for cross validation
polynomial_degree = 9

print(f"Dataset size: {total_samples} samples")
print(f"Using {polynomial_degree}-degree polynomial features")

# Define range of lambda values to test
# Note: using both log scale for small values and linear for larger ones
lambda_candidates = np.concatenate([
    np.logspace(-8, -1, 30),  # tiny lambdas from 1e-8 to 0.1
    np.linspace(0.1, 1, 20)   # larger lambdas from 0.1 to 1
])

print(f"Testing {len(lambda_candidates)} different lambda values...")

# Randomize data order (important for fair CV)
np.random.seed(42)  # for reproducible results
shuffled_indices = np.random.permutation(total_samples)
samples_per_fold = total_samples // cv_folds

# Store results for each lambda
validation_mse_results = []

print("Starting cross-validation...")

# Test each lambda value
for lambda_idx, current_lambda in enumerate(lambda_candidates):
    if lambda_idx % 10 == 0:  # progress update
        print(f"  Testing lambda {lambda_idx+1}/{len(lambda_candidates)}: {current_lambda:.6f}")
    
    # Store MSE for each fold with this lambda
    fold_mse_values = []
    
    # 5-fold cross validation
    for fold_num in range(cv_folds):
        # Define validation set for this fold
        val_start_idx = fold_num * samples_per_fold
        if fold_num == cv_folds - 1:  # last fold gets remaining samples
            val_end_idx = total_samples
        else:
            val_end_idx = (fold_num + 1) * samples_per_fold
        
        validation_indices = shuffled_indices[val_start_idx:val_end_idx]
        
        # Training set is everything else
        training_indices = np.concatenate([
            shuffled_indices[:val_start_idx], 
            shuffled_indices[val_end_idx:]
        ])
        
        # Extract training and validation data
        x_train = x_values[training_indices]
        y_train = y_values[training_indices]
        x_validation = x_values[validation_indices]
        y_validation = y_values[validation_indices]
        
        # Create polynomial feature matrices
        # Training features: [1, x, x^2, ..., x^9]
        train_feature_matrix = np.column_stack([x_train**power for power in range(polynomial_degree + 1)])
        
        # Validation features
        val_feature_matrix = np.column_stack([x_validation**power for power in range(polynomial_degree + 1)])
        
        # Ridge regression solution: w = (Phi^T * Phi + lambda * I)^(-1) * Phi^T * y
        gram_matrix = train_feature_matrix.T @ train_feature_matrix
        identity_matrix = np.eye(polynomial_degree + 1)
        regularized_gram = gram_matrix + current_lambda * identity_matrix
        rhs_vector = train_feature_matrix.T @ y_train
        
        weights = np.linalg.solve(regularized_gram, rhs_vector)
        
        # Make predictions and calculate MSE
        predictions = val_feature_matrix @ weights
        mse = np.mean((y_validation - predictions) ** 2)
        fold_mse_values.append(mse)
    
    # Average MSE across all folds for this lambda
    avg_mse_for_lambda = np.mean(fold_mse_values)
    validation_mse_results.append(avg_mse_for_lambda)

# Convert to numpy array for easier manipulation
validation_mse_results = np.array(validation_mse_results)

# Find the best lambda (lowest validation error)
optimal_lambda_index = np.argmin(validation_mse_results)
optimal_lambda = lambda_candidates[optimal_lambda_index]
best_mse = validation_mse_results[optimal_lambda_index]

print(f"\n=== CROSS-VALIDATION RESULTS ===")
print(f"Optimal lambda: {optimal_lambda:.8f}")
print(f"Best validation MSE: {best_mse:.6f}")

# Create visualization
print("\nGenerating validation curve plot...")
plt.figure(figsize=(12, 7))

# Main curve
plt.plot(lambda_candidates, validation_mse_results, 'b-', linewidth=2, 
         label='Average Validation MSE', alpha=0.8)

# Highlight the best point
plt.scatter([optimal_lambda], [best_mse], 
           color='red', s=200, marker='*', zorder=5, 
           label=f'Optimal λ = {optimal_lambda:.6f}')

# Formatting
plt.xscale('log')
plt.xlabel('Regularization Parameter (λ)', fontsize=14, fontweight='bold')
plt.ylabel('Average Validation MSE', fontsize=14, fontweight='bold')
plt.title('Ridge Regression: Finding Optimal Lambda', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot
output_filename = 'ridge_validation_curve.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Validation curve saved as: {output_filename}")
plt.show()

# Now train final models using ALL the data
print("\n=== TRAINING FINAL MODELS ===")
print("Training models on full dataset...")

# Create feature matrix for all data
full_feature_matrix = np.column_stack([x_values**power for power in range(polynomial_degree + 1)])
identity_for_ridge = np.eye(polynomial_degree + 1)

# 1. Ordinary Least Squares (no regularization)
print("Training OLS model (λ = 0)...")
ols_weights = np.linalg.solve(
    full_feature_matrix.T @ full_feature_matrix, 
    full_feature_matrix.T @ y_values
)
np.save('w_ols.npy', ols_weights)

# 2. Ridge with optimal lambda
print(f"Training Ridge model with optimal λ = {optimal_lambda:.6f}...")
ridge_weights = np.linalg.solve(
    full_feature_matrix.T @ full_feature_matrix + optimal_lambda * identity_for_ridge,
    full_feature_matrix.T @ y_values
)
np.save('w_ridge_best.npy', ridge_weights)
np.save('best_lambda.npy', optimal_lambda)

# 3. Five separate OLS models (one trained on each fold's training data)
print("Training individual OLS models for each fold...")
for fold_number in range(cv_folds):
    val_start = fold_number * samples_per_fold
    if fold_number == cv_folds - 1:
        val_end = total_samples
    else:
        val_end = (fold_number + 1) * samples_per_fold
    
    # Get training data for this fold (everything except validation)
    fold_train_indices = np.concatenate([
        shuffled_indices[:val_start], 
        shuffled_indices[val_end:]
    ])
    
    x_fold_train = x_values[fold_train_indices]
    y_fold_train = y_values[fold_train_indices]
    
    # Create feature matrix and solve
    fold_features = np.column_stack([x_fold_train**power for power in range(polynomial_degree + 1)])
    fold_weights = np.linalg.solve(
        fold_features.T @ fold_features, 
        fold_features.T @ y_fold_train
    )
    
    # Save this fold's model
    fold_filename = f'w_ols_fold{fold_number + 1}.npy'
    np.save(fold_filename, fold_weights)
    print(f"  Saved: {fold_filename}")

print(f"\n=== FILES GENERATED ===")
print(f"✓ w_ols.npy - OLS weights (λ=0, trained on all data)")
print(f"✓ w_ridge_best.npy - Ridge weights (λ={optimal_lambda:.6f}, trained on all data)")
print(f"✓ best_lambda.npy - Optimal lambda value")
print(f"✓ w_ols_fold1.npy through w_ols_fold{cv_folds}.npy - Individual fold models")
print(f"✓ {output_filename} - Validation curve plot")

print("\nDone! 🎉")