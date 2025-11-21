import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=20):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, net):
        return 1 if net >= 0 else -1    # binary step activation

    def train(self, X, T):
        for _ in range(self.epochs):
            for x, t in zip(X, T):
                net = np.dot(x, self.weights) + self.bias
                y = self.activation(net)
                
                # Perceptron Learning Rule
                error = t - y
                self.weights += self.lr * error * x
                self.bias += self.lr * error

        print("weights",self.weights)
        print("bias", self.bias)
                
       

    def predict(self, X):
        outputs = []
        for x in X:
            net = np.dot(x, self.weights) + self.bias
            outputs.append(self.activation(net))
        return np.array(outputs)


# -----------------------------
# Example: AND Gate
# -----------------------------

X = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

T = np.array([1, -1, -1, -1])  # AND gate

p = Perceptron(input_size=2, learning_rate=0.2, epochs=15)
p.train(X, T)

print("Learned Weights:", p.weights)
print("Learned Bias:", p.bias)

# Test
pred = p.predict(X)
print("Predictions:", pred)
print("Targets:", T)
