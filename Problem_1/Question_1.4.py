import numpy as np

# The problem is asking if phi^T * phi is invertible
# Then we solve for the equation and find w by usuing numpy.linalg.pinv

# Objective:
# 1. Find out if phi is inveritble by np.linalg.det
# 2. If it is, then solve the equation to find w

def get_data_matrix():
    return np.array(
        [[1,3,1,1],
        [1,5,0,2],
        [1,7,1,3],
        [1,9,0,4]])

def get_y_vector():
    return np.array([13,17,27,31])

def check_if_phi_is_invertible(phi):
    determinant = np.linalg.det(phi.T @ phi) # Computes determinant and stores into determinant variable
    # if determinant is 0, that means the matrix is not invertible 
    if(determinant == 0.0):
        print(f"The determinant is {determinant}, so the matrix IS NOT invertible.")
    else:
        print(f"The determinant is {determinant}, so the matrix IS invertible.")

def solve_psudo_inverse(phi):
    y = get_y_vector()
    w = np.linalg.pinv(phi) @ y
    print("w = ", w)
    y_pred = phi @ w
    print("Predicted y:", y_pred)
    print("Actual y:", y)
    print("Error:", np.linalg.norm(y-y_pred))


def solve_spectral_decomposition(phi):
    u, s, vh = np.linalg.svd(phi)
    y = get_y_vector()
    print("u:", u)
    print("s:", s)
    print("vh:", vh)
    threshold = 1e-10
    s_inv = np.zeros_like(s)
    for i in range(len(s)):
        if s[i] > threshold:
            s_inv[i] = 1.0 / s[i]
        else:
            s_inv[i] = 0.0
    print("s_inv:", s_inv)
    s_inv_matrix = np.diag(s_inv)
    w_svd = vh.T @ s_inv_matrix @ u.T @ y
    print("w (from SVD) =", w_svd)


if __name__ == '__main__':
    phi = get_data_matrix()
    solve_spectral_decomposition(phi)
    