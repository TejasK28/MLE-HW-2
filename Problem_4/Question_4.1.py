# IMPORTS
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
train_folder = "train"

def get_gifs_array_grayscale():

    gifs = []
    for filename in os.listdir(train_folder):
            image_path = os.path.join(train_folder, filename)
            img = Image.open(image_path).convert("L")
            gifs.append(np.array(img).flatten())

    print(f"There are {len(gifs)} gifs.")
    return gifs


def compute_mean_and_cov(gifs):
    mean = np.mean(gifs, axis=0)
    centered = gifs - mean
    cov = np.cov(centered, rowvar=False)
    return [mean, cov]
    

if __name__ == "__main__":
    gifs = get_gifs_array_grayscale()
    mean, cov = compute_mean_and_cov(gifs)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    h, w = Image.open(os.path.join(train_folder, os.listdir(train_folder)[0])).size
    plt.subplot(1, 3, 1)
    plt.imshow(mean.reshape((w, h)), cmap="gray")
    plt.title("Mean Face")
    plt.show()       