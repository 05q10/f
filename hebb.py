import numpy as np

def hebb_train(inputs, targets):
    """
    Hebbian learning for bipolar inputs and targets.
    inputs: 2D numpy array (patterns × features)
    targets: 1D array of bipolar outputs (+1, -1)
    """
    # number of features from input shape
    weights = np.zeros(inputs.shape[1])
    bias = 0

    for x, t in zip(inputs, targets):
        print(f"\nInput: {x},  Target: {t}")
        weights += x * t       # Hebbian weight update
        bias += t              # Hebbian bias update
        print("Weights:", weights)
        print("Bias:", bias)
    return weights, bias


def hebb_predict(inputs, weights, bias):
    """
    Predict output using Hebb Net.
    Activation: sign(net)
    """
    net = np.dot(inputs, weights) + bias
    return np.where(net >= 0, 1, -1)


# -----------------------------
# Example: Implement OR Gate
# Bipolar inputs: 0→-1, 1→+1
# -----------------------------

# Truth table for OR gate (bipolar)
# inputs = np.array([
#     [-1, -1],
#     [-1, +1],
#     [+1, -1],
#     [+1, +1]
# ])
inputs = np.array([
    [ +1, +1, +1,
      -1, +1, -1,
      +1, +1, +1 ],    # Pattern 'I'

    [ +1, +1, +1,
      +1, -1, +1,
      +1, +1, +1 ]     # Pattern 'O'
])
#  [ +1, +1, +1,
#       -1, +1, -1,
#       +1, +1, +1 ],    # Pattern 'I'

#     [ +1, +1, +1,
#       +1, -1, +1,
#       +1, +1, +1 ]  ,    # Pattern 'T'

#targets = np.array([-1, +1, +1, +1])   # OR in bipolar
# AND : targets = [-1, -1, -1, +1]
# NAND : targets = [+1, +1, +1, -1]
# NOR : targets = [+1, -1, -1, -1]
targets = np.array([
    +1,   # 'I' is member of class
    -1    # 'O' is NOT member of class
])


# Train Hebb Net
weights, bias = hebb_train(inputs, targets)

print("Learned Weights:", weights)
print("Learned Bias:", bias)

# Test
pred = hebb_predict(inputs, weights, bias)
print("Predictions:", pred)
print("Targets:", targets)
