import numpy as np
from PIL import Image 
import os
import glob
import matplotlib.pyplot as plt

# LOAD
train_folder_name = "train"
train_files = glob.glob(os.path.join(train_folder_name, "*"))
train_files = [f for f in train_files if os.path.isfile(f)]

# Print files
for file in train_files:
    print(file)
print(f"\nFound {len(train_files)} training images")

# Load all training faces
faces = []
for filepath in sorted(train_files):
    try:
        img = Image.open(filepath).convert('L')  # Grayscale
        img_array = np.array(img, dtype=np.float64)  # Convert to float
        flat_face = img_array.flatten()  # Flatten to 1D
        faces.append(flat_face)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")

# Convert to numpy array
X = np.array(faces)  # Shape: (N, d)
N, d = X.shape

print(f"\nLoaded data matrix X:")
print(f"  Shape: {X.shape}")
print(f"  {N} training faces")
print(f"  {d} pixels per face")

# Get image dimensions (from first image)
first_img = Image.open(train_files[0]).convert('L')
img_height, img_width = np.array(first_img).shape
print(f"  Image size: {img_width} x {img_height}")


# Compute E[x] which is the mean face

x_bar = np.mean(X, axis=0)  # Average across all faces

print(f"\nMean face E[X] computed")
print(f"  Shape: {x_bar.shape}")
print(f"  Mean pixel value: {np.mean(x_bar):.2f}")
print(f"  Min: {np.min(x_bar):.2f}, Max: {np.max(x_bar):.2f}")


# Center the data

X_centered = X - x_bar

print(f"\nData centered")
print(f"  Centered mean: {np.mean(X_centered):.10f} (should be ≈ 0)")

# SVD

print("Performing SVD on X_centered.T ...")
# Use SVD: X_centered.T = U S V^T
# Then COV = (1/N) * X_centered.T @ X_centered = U @ diag(S^2/N) @ U.T

U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)

# Compute eigenvalues from singular values
eigenvalues = (S ** 2) / N
eigenvectors = U  # These are the eigenfaces

print(f"\nSpectral decomposition complete!")
print(f"  Eigenvectors (E) shape: {eigenvectors.shape}")
print(f"  Eigenvalues (Λ) shape: {eigenvalues.shape}")
print(f"  Number of eigenfaces: {len(eigenvalues)}")