import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

class FaceAnalyzer:
    def __init__(self, train_folder="train"):
        self.train_folder = train_folder
        self.training_images = None
        self.mean_image = None
        self.X_centered = None
        self.cov_matrix = None
        self.eigenvals = None
        self.eigenvecs = None
        
    def load_training_data(self):
        """
        Load and preprocess training images from the specified folder
        """
        print("Loading training images...")
        image_data = []
        
        filenames = sorted(os.listdir(self.train_folder))
        for filename in filenames:
            img_path = os.path.join(self.train_folder, filename)
            image = Image.open(img_path)
            img_array = np.array(image)
            image_data.append(img_array.flatten())
        
        self.training_images = np.array(image_data)
        print(f"Loaded {len(self.training_images)} training images")
        
    def compute_pca(self):
        """
        Compute PCA components: mean, covariance, eigenvalues and eigenvectors
        """
        if self.training_images is None:
            raise ValueError("Training data not loaded. Call load_training_data() first.")
            
        print("Computing E[X] (mean image)...")
        self.mean_image = np.mean(self.training_images, axis=0)
        
        print("Computing centered data...")
        self.X_centered = self.training_images - self.mean_image
        
        print("Computing covariance matrix...")
        # Using d×d covariance matrix formulation
        num_samples = len(self.training_images)
        self.cov_matrix = (1 / (num_samples - 1)) * self.X_centered.T @ self.X_centered
        
        print("Computing spectral decomposition...")
        self.eigenvals, self.eigenvecs = np.linalg.eigh(self.cov_matrix)
        
        # Sort in descending order of eigenvalues
        sorted_idx = np.argsort(self.eigenvals)[::-1]
        self.eigenvecs = self.eigenvecs[:, sorted_idx]
        self.eigenvals = self.eigenvals[sorted_idx]
        
        print(f"Shape of COV(X,X): {self.cov_matrix.shape}")
        print(f"Shape of E[X]: {self.mean_image.shape}")
        
    def display_mean_face(self, img_height=60):
        """
        Display the computed mean face
        """
        if self.mean_image is None:
            raise ValueError("PCA not computed. Call compute_pca() first.")
            
        plt.figure(figsize=(6, 4))
        plt.imshow(self.mean_image.reshape(img_height, -1), cmap="gray")
        plt.title("E[X] - Mean Face")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
    def reconstruct_image(self, test_image_path, num_components_list=[2, 10, 100, 1000, 4000], img_height=60):
        """
        Reconstruct a test image using different numbers of principal components
        """
        if self.eigenvecs is None:
            raise ValueError("PCA not computed. Call compute_pca() first.")
            
        # Load test image
        test_img = Image.open(test_image_path)
        test_vec = np.array(test_img).flatten()
        test_filename = os.path.basename(test_image_path)
        
        # Create subplot layout
        fig, axes = plt.subplots(1, len(num_components_list) + 1, figsize=(24, 4))
        
        # Show original image
        axes[0].imshow(test_vec.reshape(img_height, -1), cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        # Reconstruct with different M values
        for i, m in enumerate(num_components_list):
            print(f"Reconstructing with M={m}...")
            
            # Select top M eigenvectors
            top_eigenvecs = self.eigenvecs[:, :m]
            
            # Center the test image
            test_centered = test_vec - self.mean_image
            
            # Reconstruction formula: x̃ ≈ x̄ + E_M * E_M^T * (x - x̄)
            projection_coeffs = top_eigenvecs.T @ test_centered
            reconstructed = self.mean_image + top_eigenvecs @ projection_coeffs
            
            # Display reconstruction
            axes[i+1].imshow(reconstructed.reshape(img_height, -1), cmap="gray")
            axes[i+1].set_title(f"M={m}")
            axes[i+1].axis("off")
        
        plt.suptitle(f"Reconstruction with Different M Values ({test_filename})")
        plt.tight_layout()
        plt.show()
        
    def display_eigenfaces(self, num_faces=10, img_height=60):
        """
        Display the top eigenfaces (principal components)
        """
        if self.eigenvecs is None:
            raise ValueError("PCA not computed. Call compute_pca() first.")
            
        print("Displaying eigenfaces...")
        
        rows = 2
        cols = (num_faces + 1) // 2  # Ceiling division to handle odd numbers
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 8))
        axes = axes.flatten()
        
        for i in range(num_faces):
            # Get the i-th eigenface
            eigenface = self.eigenvecs[:, i]
            
            # Normalize eigenface for better visualization
            eigenface_min = eigenface.min()
            eigenface_max = eigenface.max()
            eigenface_norm = (eigenface - eigenface_min) / (eigenface_max - eigenface_min) * 255
            
            # Display eigenface
            axes[i].imshow(eigenface_norm.reshape(img_height, -1), cmap="gray")
            axes[i].set_title(f"Eigenface {i+1}\n(λ={self.eigenvals[i]:.0f})")
            axes[i].axis("off")
        
        # Hide unused subplots if any
        for j in range(num_faces, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle(f"Top {num_faces} Eigenfaces (Principal Components)")
        plt.tight_layout()
        plt.show()

def run_face_analysis(train_folder="train", test_folder="test"):
    # Initialize analyzer
    analyzer = FaceAnalyzer(train_folder)
    
    # Load data and compute PCA
    analyzer.load_training_data()
    analyzer.compute_pca()
    
    # Display mean face
    analyzer.display_mean_face()
    
    # Reconstruct a test image
    test_files = sorted(os.listdir(test_folder))
    if test_files:
        test_image_path = os.path.join(test_folder, test_files[0])
        analyzer.reconstruct_image(test_image_path)
    else:
        print("No test images found!")
    
    # Display eigenfaces
    analyzer.display_eigenfaces()
    
    print("Analysis complete!")
    
    return analyzer

if __name__ == "__main__":
    face_analyzer = run_face_analysis()