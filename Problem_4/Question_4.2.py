import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

train_folder = "train"
test_folder = "test"

# Load and flatten images
def load_images(folder):
    images = []
    for f in sorted(os.listdir(folder)):
        img = Image.open(os.path.join(folder, f)).convert("L")
        images.append(np.array(img).flatten())
    return np.array(images)

# Load data
train = load_images(train_folder)
test = load_images(test_folder)

# Compute mean and eigenvectors
mean = np.mean(train, axis=0)
centered = train - mean
cov = np.cov(centered, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]

# Get image shape (FIXED: PIL returns (width, height))
img = Image.open(os.path.join(train_folder, sorted(os.listdir(train_folder))[0]))
w, h = img.size  # PIL gives (width, height)
img_shape = (h, w)  # NumPy uses (height, width)

# Approximate with different M values
M_values = [2, 10, 100, 1000, 4000]
test_img = test[0]

fig, axes = plt.subplots(1, len(M_values), figsize=(15, 3))

for i, M in enumerate(M_values):
    # Reconstruct: x̄ + E_M * E_M^T * (x - x̄)
    E_M = eigvecs[:, :M]
    approx = mean + E_M @ (E_M.T @ (test_img - mean))
    
    axes[i].imshow(approx.reshape(img_shape), cmap='gray')
    axes[i].set_title(f'M = {M}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('eigenface_approximations.png', dpi=150, bbox_inches='tight')
plt.show()