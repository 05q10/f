import numpy as np

class KohonenSingleStep:
    def __init__(self, W):
        """Initialize SOM with fixed textbook weight matrix."""
        # W is given in row format, convert to (5 × 2) matrix
        self.weights = np.array(W)        # shape: (2,5)
        self.weights = self.weights.T     # convert → (5,2)
        self.num_clusters = self.weights.shape[1]

    def squared_distance(self, x, w):
        return np.sum((x - w)**2)

    def winner(self, x):
        distances = [self.squared_distance(x, self.weights[:, j]) 
                     for j in range(self.num_clusters)]
        print("Squared distances:", distances)
        return np.argmin(distances)

    def update(self, x, lr):
        winner = self.winner(x)
        print(f"\nWinner Cluster Unit: Y{winner+1}")

        # Update winning column
        self.weights[:, winner] += lr * (x - self.weights[:, winner])

        print("\nUpdated Weight Matrix (column format):")
        print(self.weights)

        return self.weights




W = [
    [1.0, 0.9, 0.7, 0.5, 0.3],   # w1  (row form)
    [0.3, 0.5, 0.7, 0.9, 1.0]    # w2
]

x = np.array([0.0, 0.5, 1.0, 0.5, 0.0])  # input
lr = 0.25                                  # learning rate

som = KohonenSingleStep(W)
updated_matrix = som.update(x, lr)
