import numpy as np

class DiscreteHopfieldNetwork:
    def __init__(self, pattern):
        """
        Initializes and trains a Hopfield network for a single pattern.
        
        Args:
            pattern (np.array): The bipolar pattern vector to store.
        """
        self.pattern = np.array(pattern)
        self.n = len(pattern)
        self.W = self._create_weight_matrix()

    def _create_weight_matrix(self):
        """
        Computes the weight matrix using the Hebbian learning rule.
        W = p * p^T, with diagonal elements set to zero.
        """
        # Reshape pattern to a column vector for outer product
        p = self.pattern.reshape(self.n, 1)
        
        # Hebbian rule: outer product of the pattern with itself
        W = p @ p.T
        
        # Set diagonal elements to zero (no self-connections)
        np.fill_diagonal(W, 0)
        
        return W

    def recall(self, noisy_pattern, max_iters=10):
        """
        Attempts to recall the original pattern from a noisy input.
        
        Args:
            noisy_pattern (np.array): The input vector with noise or missing parts.
            max_iters (int): The maximum number of iterations for convergence.
        
        Returns:
            np.array: The final, stable state of the network.
        """
        s = np.array(noisy_pattern).copy()
        print(f"Initial (noisy) state: {s}")

        for i in range(max_iters):
            print(f"\n--- Iteration {i+1} ---")
            s_old = s.copy()
            
            # Asynchronous update: update each neuron one by one
            for j in range(self.n):
                net_input = np.dot(self.W[j, :], s)
                s[j] = 1 if net_input >= 0 else -1
                print(f"Updating neuron {j+1}: net_input={net_input:.2f} -> new state={s}")

            # Check for convergence
            if np.array_equal(s, s_old):
                print("\nNetwork has converged to a stable state.")
                break
        
        return s

# 1. Define the input vector to be stored
stored_pattern = np.array([1, 1, 1, -1])

# 2. Create and train the Hopfield network
net = DiscreteHopfieldNetwork(stored_pattern)
print("Stored Pattern:", net.pattern)
print("Calculated Weight Matrix (W):\n", net.W)

# 3. Test with a vector where the first two components are missing (set to 0)
test_pattern = np.array([0, 0, 1, -1])

print("\n--- Testing Recall ---")
recalled_pattern = net.recall(test_pattern)

print("\n--- Final Result ---")
print(f"Original Stored Pattern: {net.pattern}")
print(f"Noisy Input Pattern:     {test_pattern}")
print(f"Recalled Pattern:        {recalled_pattern}")

if np.array_equal(net.pattern, recalled_pattern):
    print("\nSuccess: The network successfully reconstructed the original pattern.")
else:
    print("\nFailure: The network did not reconstruct the original pattern.")
