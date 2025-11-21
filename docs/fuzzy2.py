import numpy as np

R=np.array([[0.6,0.3],[0.2,0.9]])
S=np.array([[1.0,0.5,0.3],[0.8,0.4,0.7]])

def max_min(R,S):
    n_x,n_y=R.shape
    n_y,n_z=S.shape
    T = np.zeros((n_x, n_z))
    for i in range (n_x):
        for k in range (n_z):
            values=[]
            for j in range (n_y):
                values.append(min(R[i,j],S[j,k]))
            T[i,k]=max(values)

    return T

def max_product(R,S):
    n_x,n_y=R.shape
    n_y,n_z=S.shape
    T = np.zeros((n_x, n_z))
    for i in range (n_x):
        for k in range (n_z):
            values=[]
            for j in range (n_y):
                values.append((R[i,j]*S[j,k]))
            T[i,k]=max(values)

    return T


T_maxmin = max_min(R, S)
print("\n=== MAX–MIN COMPOSITION R ○ S ===")
print(T_maxmin)

T_maxprod = max_product(R,S)
print("\n=== MAX–PRODUCT COMPOSITION R ○ S ===")
print(T_maxprod)