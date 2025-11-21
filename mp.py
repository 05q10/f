def mp_neuron(inputs, weights, threshold):
    """
    Implements a McCulloch-Pitts (MP) Neuron
    inputs: list of binary inputs (0/1)
    weights: list of weights
    threshold: activation threshold
    for AND
    inputs = [1,1]
    weights = [1,1]
    threshold = 2
     for OR
     inputs = [1,0]
    weights = [1,1]
    threshold = 1
    for NAND
    threshold = 1.5
    weights = [-1, -1]



    """
    
    # Weighted sum
    net = sum(i * w for i, w in zip(inputs, weights))
    
    # Activation function (Step Function)
    output = 1 if net >= threshold else 0
    
    return net, output


# Example Usage
inputs = [1, 0, 1]          # You can change inputs here
weights = [1, 1, 1]         # Equal weights for simple logic
threshold = 2               # Adjust threshold to change logic

net, output = mp_neuron(inputs, weights, threshold)

print("Inputs:", inputs)
print("Weights:", weights)
print("Threshold:", threshold)
print("Weighted Sum:", net)
print("Neuron Output:", output)


