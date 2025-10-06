import numpy as np

# Q1.3

# Steps
# 1. Calculate phi^t * phi
# 2. Check determinant 
# 3. Use psudo-inverse to solve


# Make the data matrix
data_matrix = np.array([
    [1, 3, 1, 1],
    [1, 5, 0, 2],
    [1, 7, 1, 3],
    [1, 9, 0, 4]
])

y = np.array([13,17,27,31])

print("The data matrix:\n", data_matrix,"\n===============")


# Mulitply the matrixies
product = data_matrix.T @ data_matrix

print("phi^T * phi:\n", product)

# Check determinant
det = np.linalg.det(product)
print("Determinant of Φᵗ Φ:", det,"\n===============")

if(det <= 0.0):
    print("Determinant is 0, so the data matrix is NOT invertible\n===============")
else:
    print("The data matrix is invertible")


# Calculate Φᵗ y
phi_t_y = data_matrix.T @ y
print("Φᵗ y:", phi_t_y,"\n===============")

inverse = np.linalg.pinv(product)

print("The inverse is:\n", inverse, "\n===============")

w = inverse @ phi_t_y
print("The w vector is:", w, "\n===============")

print("Solution w̄:")
print(f"w₀ = {w[0]}")
print(f"w₁ = {w[1]}")
print(f"w₂ = {w[2]}")
print(f"w₃ = {w[3]}")
print("\n===============\n")
