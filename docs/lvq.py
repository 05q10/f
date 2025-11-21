# ---------- LVQ Implementation ----------
import numpy as np

# Input vectors
X = np.array([
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1]
], dtype=float)

# Class labels for each input
T = np.array([1, 1, 2, 2])          # example classes

# Initial weight matrix (same as SOM)
W = np.array([
    [0.2, 0.9],
    [0.4, 0.7],
    [0.6, 0.5],
    [0.8, 0.3]
], dtype=float)

# Class label for each weight vector
C = np.array([1, 2])                # neuron 1 → class 1, neuron 2 → class 2

alpha = 0.5
num_clusters = 2

def euclidean_distance(w, x):
    return np.sum((w - x)**2)

print("Initial weights:\n", W)

# -------- LVQ Training Loop --------
for idx, x in enumerate(X, start=1):
    print(f"\n### Input {idx}: {x}  (Class = {T[idx-1]}) ###")

    # Step 1: Distance to clusters
    distances = []
    for j in range(num_clusters):
        d = euclidean_distance(W[:, j], x)
        distances.append(d)
        print(f"D({j+1}) = {d:.4f}")

    # Step 2: Winner neuron
    winner = np.argmin(distances)
    print(f"Winner = Y{winner+1}  (Class = {C[winner]})")

    # Step 3: LVQ update
    if C[winner] == T[idx - 1]:
        print("Correct → move weights closer")
        W[:, winner] = W[:, winner] + alpha * (x - W[:, winner])
    else:
        print("Wrong → push weights away")
        W[:, winner] = W[:, winner] - alpha * (x - W[:, winner])

    print("Updated W:\n", W)

print("\nFinal LVQ Weights:\n", W)