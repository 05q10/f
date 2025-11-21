import numpy as np

class HopfieldNetwork:
    def __init__(self):
        self.W = None       # Weight matrix

    # -------------------------------
    # TRAINING USING HEBB RULE
    # -------------------------------
    def train(self, patterns):
        """Train Hopfield Net using Hebbian learning.
           patterns: list/array of bipolar vectors (+1, -1)
        """
        patterns = np.array(patterns)
        n = patterns.shape[1]

        # Initialize weight matrix
        self.W = np.zeros((n, n))

        # Hebbian learning: W = Σ(p^T * p)
        for p in patterns:
            p = p.reshape(n, 1)
            self.W += np.dot(p, p.T)

        # Zero out diagonal (no self-connection)
        np.fill_diagonal(self.W, 0)

        # Symmetric weights (W = W^T)
        self.W = (self.W + self.W.T) / 2

    # -------------------------------
    # RECALL / STATE UPDATE
    # -------------------------------
    def recall(self, pattern, steps=5, synchronous=False):
        """Recall stored pattern from input."""
        pattern = pattern.copy()

        for _ in range(steps):

            if synchronous:
                net = np.dot(self.W, pattern)
                pattern = np.where(net >= 0, 1, -1)

            else:  # asynchronous update
                for i in range(len(pattern)):
                    net = np.dot(self.W[i], pattern)
                    pattern[i] = 1 if net >= 0 else -1

        return pattern


# ------------------------------------------------
# Example Usage
# ------------------------------------------------

# Training patterns (bipolar)
patterns = [
    [1, -1, 1, -1],
    [1, 1, -1, -1]
]

hop = HopfieldNetwork()
hop.train(patterns)

print("Learned Weight Matrix:\n", hop.W)

# Recall from a noisy pattern
test = np.array([1, -1, -1, -1])
print("\nTest Input:", test)

output = hop.recall(test, steps=10)
print("Recalled Output:", output)
