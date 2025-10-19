import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Define the training folder path
training_data_path = "train"

# Function to load and preprocess images from a directory
def load_and_flatten_images(folder_path):
    image_vectors = []
    file_list = sorted(os.listdir(folder_path))  # Sort to ensure consistent ordering
    
    for filename in file_list:
        # Open image and convert to grayscale
        img_path = os.path.join(folder_path, filename)
        image = Image.open(img_path).convert("L")
        
        # Flatten the image into a 1D array
        flattened_img = np.array(image).flatten()
        image_vectors.append(flattened_img)
    
    return np.array(image_vectors)

# Load the training dataset
print("Loading training images...")
training_data = load_and_flatten_images(training_data_path)

# Calculate the mean face (average of all training images)
mean_face = np.mean(training_data, axis=0)

# Center the data by subtracting the mean
centered_data = training_data - mean_face

# Compute covariance matrix - this might take a while for large datasets
print("Computing covariance matrix...")
covariance_matrix = np.cov(centered_data, rowvar=False)

# Find eigenvalues and eigenvectors
print("Computing eigenvalues and eigenvectors...")
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Sort eigenvalues and eigenvectors in descending order
# (Most important components first)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Get the dimensions of original images for reshaping
sample_img_path = os.path.join(training_data_path, sorted(os.listdir(training_data_path))[10])
sample_image = Image.open(sample_img_path)
image_width, image_height = sample_image.size
original_shape = (image_height, image_width)  # Note: PIL uses (width, height), numpy uses (height, width)

# Create visualization of the top 10 eigenfaces
print("Generating eigenface visualization...")
fig, subplot_axes = plt.subplots(2, 5, figsize=(15, 6))
subplot_axes = subplot_axes.flatten()

for i in range(10):
    # Reshape eigenvector back to image dimensions
    eigenface = eigenvectors[:, i].reshape(original_shape)
    
    # Normalize eigenface for better visualization
    # Scale values to 0-255 range for proper display
    min_val = eigenface.min()
    max_val = eigenface.max()
    eigenface_display = (eigenface - min_val) / (max_val - min_val) * 255
    
    # Display the eigenface
    subplot_axes[i].imshow(eigenface_display, cmap='gray')
    subplot_axes[i].set_title(f'Eigenvalue = {eigenvalues[i]:.2f}')
    subplot_axes[i].axis('off')  # Hide axes for cleaner look

plt.tight_layout()
plt.savefig('top_10_eigenfaces.png', dpi=150, bbox_inches='tight')
plt.show()

# Print out the top eigenvalues for reference
print("\nTop 10 eigenvalues (in descending order):")
for idx in range(10):
    print(f"  Eigenvalue {idx+1}: {eigenvalues[idx]:.2f}")