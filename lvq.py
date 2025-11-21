import numpy as np

# Training data 
X = np.array([
    [0, 0, 1, 1],  
    [1, 0, 0, 0],  
    [0, 0, 0, 1],  
    [1, 1, 0, 0],  
    [0, 1, 1, 0]   
])
T = np.array([1, 2, 2, 1, 1])  

# Step 0: Initialize reference weights (first 2 vectors as initial weights)
W = np.array([
    [0, 0, 1, 1],  # Class 1 (w1)
    [1, 0, 0, 0]   # Class 2 (w2)
], dtype=float)

classes = [1, 2]  # Assign cluster classes
alpha = 0.1       # Learning rate

def euclidean_distance(x, w):
    return np.sum((x - w) ** 2)

# Training using the remaining 3 input vectors
training_vectors = [(X[2], T[2]), (X[3], T[3]), (X[4], T[4])]

for x, target in training_vectors:
    # Step 3: Find winning unit
    distances = [euclidean_distance(x, w) for w in W]
    J = np.argmin(distances)  # index of winner

    # Step 4: Update rule
    if classes[J] == target:  # Same class → move closer
        W[J] = W[J] + alpha * (x - W[J])
    else:  # Different class → move away
        W[J] = W[J] - alpha * (x - W[J])

    # Show update
    print(f"Input: {x}, Target={target}, Winner={J+1}, Updated Weights:\n{W}\n")
