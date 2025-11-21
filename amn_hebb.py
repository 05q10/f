import numpy as np

def hebb_associative_memory(X, Y):
    """
    Hebbian learning for Associative Memory Network (AMN)
    X : input patterns (n patterns × n inputs)
    Y : target output patterns (n patterns × m outputs)
    Returns W : weight matrix (input_dim × output_dim)
    """

    n_samples, n_inputs = X.shape
    _, n_outputs = Y.shape

    # Step 0: Initialize weight matrix to zero
    W = np.zeros((n_inputs, n_outputs))

    # Step 1: For each training pair
    for s in range(n_samples):
        xs = X[s]         # Step 2: input layer activation
        yt = Y[s]         # Step 3: output layer activation

        # Step 4: Hebbian weight update: W = W + x * y^T (outer product)
        W += np.outer(xs, yt)

    return W


def recall(W, x):
    """
    Recall output from associative memory: y = sign(W^T x)
    """
    y = np.dot(x, W)
    return np.where(y >= 0, 1, -1)


# --------------------------------------------
# Example: Autoassociative Memory (X -> X)
# (Can change to heteroassociative easily)
# --------------------------------------------

# Bipolar input patterns
X = np.array([
    [1, -1, 1],
    [-1, -1, 1]
])

# For autoassociative memory, Y = X
Y = X.copy()

# Train AMN
W = hebb_associative_memory(X, Y)

print("Learned Weight Matrix:\n", W)

# Test recall with noisy input
test = np.array([1, -1, -1])
pred = recall(W, test)

print("\nInput:", test)
print("Recalled Output:", pred)
