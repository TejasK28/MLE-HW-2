import numpy as np

def get_original_data():
    phi = np.array(
        [[1,3,1,1],
         [1,5,0,2],
         [1,7,1,3],
         [1,9,0,4]])
    y = np.array([13,17,27,31])
    return phi, y

def get_new_data():
    phi_new = np.array(
        [[1,11,1,5],
         [1,13,0,6],
         [1,15,1,7],
         [1,17,0,8]])
    y_new = np.array([41.04, 45.77, 55.04, 59.96])
    return phi_new, y_new

def solve():
    phi_old, y_old = get_original_data()
    phi_new, y_new = get_new_data()
    phi_combined = np.vstack([phi_old, phi_new])
    y_combined = np.hstack([y_old, y_new])
    
    print("phi_combined:", phi_combined)
    print("y_combined:", y_combined)

    det = np.linalg.det(phi_combined.T @ phi_combined)
    print(f"Determinant: {det}")
    if abs(det) < 1e-10:
        print("NOT invertible")
    else:
        print("YES invertible!")
    
    w_old = np.linalg.pinv(phi_old) @ y_old
    w_new = np.linalg.pinv(phi_combined) @ y_combined


    print("w_old:", w_old)
    print("w_new:", w_new)

if __name__ == "__main__":
    solve()