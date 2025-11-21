import numpy as np

x_vals=np.array([0.0, 0.25, 0.5, 0.75, 1.0])
A=np.array([0,1,0.5,0,0])
B=np.array([0,0.75,1,0.5,0])


A_U_B= np.maximum(A,B)
A_n_B= np.minimum(A,B)

A_comp=1-A
B_comp=1-B

Bounded_sum=np.minimum(1,A+B)
Bounded_diff=np.maximum(0,A-B)

def cartesian(A,B):
    R=np.zeros((len(A),len(B)))
    for i in range (len(A)):
        for j in range(len(B)):
            R[i,j]=min(A[i],B[j])

    return R


print("x-values:      ", x_vals)
print("μA(x):         ", A)
print("μB(x):         ", B)

print("\n(a) A ∪ B      =", A_U_B)
print("(b) A ∩ B      =", A_n_B)
print("(c) Ā         =", A_comp)
print("(d) B̄         =", B_comp)

print(f"Cartesian product : {cartesian(A,B)}")