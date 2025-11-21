#exp7-bam

import numpy as np

# ---------------------------
# Helper functions
# ---------------------------

def binary_to_bipolar(v):
    return np.where(v == 0, -1, 1)

def bipolar_step(x):
    return np.where(x > 0, 1, -1)

# ---------------------------
# Training Data (binary)
# ---------------------------

S_binary = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 1, 1, 0]
], dtype=int)

T_binary = np.array([
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1]
], dtype=int)

# Convert to bipolar
S = binary_to_bipolar(S_binary)
T = binary_to_bipolar(T_binary)

# ---------------------------
# (a) Compute Weight Matrix W = Σ s_p^T * t_p
# ---------------------------

W = np.zeros((4, 2))
for p in range(4):
    W += np.outer(S[p], T[p])

print("Weight Matrix W:\n", W)

# ---------------------------
# Function to test BAM forward pass (S → T)
# ---------------------------

def test_s_to_t(s):
    s = np.array(s).reshape(1, -1)
    y_in = s @ W
    t_out = bipolar_step(y_in)
    return y_in.flatten(), t_out.flatten()

# ---------------------------
# (b) Test original input patterns
# ---------------------------

print("\n--- Testing Original Patterns ---")
for i in range(4):
    yin, tout = test_s_to_t(S[i])
    print(f"\nInput S({i+1}) = {S[i]}")
    print("Net input to T =", yin)
    print("Output =", tout)

# ---------------------------
# (c) Testing with mistakes / missing data
# ---------------------------

noisy_tests = [
    [1, 0, -1, -1],
    [-1, 0, 0, -1],
    [-1, 1, 0, -1],
    [1, 1, -1, -1],
    [1, 1]   # test T directly
]

print("\n--- Testing Noisy / Partial Inputs ---")
for x in noisy_tests:
    if len(x) == 4:
        print(f"\nTesting noisy S input: {x}")
        yin, tout = test_s_to_t(x)
        print("Net input =", yin)
        print("Output =", tout)
    else:
        # Testing a T input (reverse mode)
        print(f"\nTesting T input: {x}")
        x = np.array(x).reshape(1, -1)
        yin = x @ W.T
        s_out = bipolar_step(yin)
        print("Recovered S =", s_out)
