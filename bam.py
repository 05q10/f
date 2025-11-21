import numpy as np

def hebb_bam_train(X, Y):
    """
    Train BAM using Hebbian outer product rule.
    X: input patterns (P × n)
    Y: output patterns (P × m)
    Returns: weight matrix W (n × m)
    """
    n_samples = X.shape[0]
    n_inputs = X.shape[1]
    n_outputs = Y.shape[1]

    W = np.zeros((n_inputs, n_outputs))

    # Hebbian outer product update for all pattern pairs
    for p in range(n_samples):
        W += np.outer(X[p], Y[p])

    return W


def bipolar_sign(x):
    """Sign activation with bipolar output."""
    return np.where(x >= 0, 1, -1)


def bam_recall(W, x_init, max_iters=10):
    """
    Perform BAM recall until convergence.
    x_init : initial input pattern
    Returns final (X, Y)
    """
    X = x_init.copy()

    for _ in range(max_iters):
        Y = bipolar_sign(np.dot(X, W))       # forward pass
        X_new = bipolar_sign(np.dot(Y, W.T)) # backward pass

        # Convergence check
        if np.array_equal(X, X_new):
            break

        X = X_new

    return X, Y


# ---------------------------------------------------------
# Example Patterns (Change these to store other associations)
# ---------------------------------------------------------
X = np.array([
    [ 1, -1,  1],     # Pattern 1 input
    [-1,  1, -1]      # Pattern 2 input
])

Y = np.array([
    [ 1,  1],         # Pattern 1 output
    [-1, -1]          # Pattern 2 output
])

# Train BAM
W = hebb_bam_train(X, Y)
print("Learned Weight Matrix (W):\n", W)

# Test BAM recall with a noisy input
x_test = np.array([1, -1, -1])
X_recalled, Y_recalled = bam_recall(W, x_test)

print("\nGiven Input:", x_test)
print("Recalled X:", X_recalled)
print("Recalled Y:", Y_recalled)
