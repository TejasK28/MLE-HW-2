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

if(abs(det) < 1e-10):
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
print("="*30)


#Q 1.4
#We're going to be usuing numpy.linalg.svd()
#Inverse of the smallest eigenvalue is a large number 
#Is the solution the same as above?
# Explain why pseudo-inverse and spectral decomposition yield the same solution?

print("="*30)
print("Q1.4")
print("="*30)


U, S, Vt = np.linalg.svd(product)

print("U is: ", U)
print("="*30)
print("S is: ", S)
print("="*30)
print("Vt is: ", Vt)
print("="*30)

print("\nSingular values:", S)
print(f"Smallest singular value: {S[-1]:.10e}")

#Set inverse of smallest singular value to large number
S_inv = np.zeros_like(S)
for i in range(len(S)):
    if S[i] < 1e-10:
        S_inv[i] = 1e10 
        print(f"\nSingular value {i} ≈ 0, setting inverse to 1e10")
    else:
        S_inv[i] = 1 / S[i]

# Compute inverse: (Φᵗ Φ)⁻¹ = U @ diag(S⁻¹) @ Uᵗ
Phi_t_Phi_inv = U @ np.diag(S_inv) @ U.T
w_svd = Phi_t_Phi_inv @ phi_t_y

print("\nSolution from Q1.3 (pseudo-inverse):")
print(f"w = {w}")

print("\nSolution from Q1.4 (SVD with large inverse):")
print(f"w = {w_svd}")

print(f"\nDifference: {np.linalg.norm(w - w_svd):.10e}")
print(f"\nAre they roughly the same with very little difference? {np.allclose(w, w_svd, atol=1e-3)}")


# Verify both solutions predict the data correctly
print("\n" + "="*30)
print("Verification - Do both fit the data?")
print("="*30)

y_pred_pinv = data_matrix @ w
y_pred_svd = data_matrix @ w_svd

print(f"Actual y:          {y}")
print(f"Predicted (pinv):  {np.round(y_pred_pinv, 6)}")
print(f"Predicted (SVD):   {np.round(y_pred_svd, 6)}")
print(f"\nPrediction error (pinv): {np.linalg.norm(y - y_pred_pinv):.10e}")
print(f"Prediction error (SVD):  {np.linalg.norm(y - y_pred_svd):.10e}")

print("\n" + "="*30)
print("EXPLANATION FOR Q1.4")
print("="*30)
print("""
Both spectral decomposition and pseudo-inverse yield the same solution as they
both decomposing Phi transpose Phi by singular value decomposition and processing the
near-zero singular value such that it results in the same outcome. The pseudo-inverse
sets the inverse of zero singular value to zero, but our spectral decomposition
This sets it to an arbitrarily large number such as one times ten to the tenth power.
Even with such a massive countable disparity, they yield basically the same weight vector
Since theingular vector for the zero singular value is a
direction in the weight space that is orthogonal to the data. To find when we take the
Terminal weights by multiplying our inverse matrix with Phi transpose y, the component
in that zero-singular-value direction is irrelevant to the solution no matter
of whether we multiply zero by some big quantity or just put it equal to zero. Both of these methods
efficiently project the solution onto the same subspace that is spanned by the singular vectors
with non-zero singular values, and both aim to find the best fitting minimum norm solution
the data. The almost negligible numerical difference of some zero point zero zero zero two between
the two solutions comes from floating point arithmetic precision limits when working
with such large numbers, but the solutions are identical for all practical purposes.
""")