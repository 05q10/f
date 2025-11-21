import numpy as np

# ----------------------------------------
# Kohonen SOM Implementation for 2 clusters
# ----------------------------------------

class KohonenSOM:
    def __init__(self, input_dim, num_clusters=2, lr=0.5, epochs=10):
        self.num_clusters = num_clusters
        self.lr = lr
        self.epochs = epochs
        
        # Initialize weights randomly between 0 and 1
        self.weights = np.array([
    [1.0, 0.9, 0.7, 0.5, 0.3],
    [0.3, 0.5, 0.7, 0.9, 1.0]
])


    def winner(self, x):
        """Return index of nearest weight vector (Euclidean distance)."""
        distances = np.linalg.norm(self.weights - x, axis=1)
        return np.argmin(distances)

    def train(self, X):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}")
            for x in X:
                w_idx = self.winner(x)
                
                # Update winner weights
                self.weights[w_idx] += self.lr * (x - self.weights[w_idx])
                
                print(f"Input: {x} → Winner: {w_idx}")
                print(f"Updated cluster {w_idx} weights: {self.weights[w_idx]}")
            
            # Decay learning rate
            self.lr *= 0.9

        return self.weights

    def classify(self, X):
        labels = []
        for x in X:
            labels.append(self.winner(x))
        return np.array(labels)


# -----------------------------
# Given vectors
# -----------------------------
X = np.array([
    [0.0, 0.5, 1.0, 0.5, 0.0]
])


som = KohonenSOM(input_dim=5, num_clusters=2, lr=0.25, epochs=1)

print("Initial Weights:\n", som.weights)

final_weights = som.train(X)

print("\nFinal Cluster Weights:\n", final_weights)

cluster_labels = som.classify(X)

