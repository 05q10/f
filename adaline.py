import numpy as np

class AdalineBipolar:
    def __init__(self, input_size, learning_rate=0.1, epochs=40):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = learning_rate
        self.epochs = epochs

    def net_input(self, x):
        return np.dot(x, self.weights) + self.bias

    def train(self, X, T):
        for epoch in range(self.epochs):
            total_error = 0

            for x, t in zip(X, T):
                yin = self.net_input(x)        # linear ADALINE output
                error = t - yin                # continuous error
                total_error += error**2        # accumulate squared error

                # LMS / Delta rule update
                self.weights += self.lr * error * x
                self.bias += self.lr * error

            print(f"Epoch {epoch+1} - Total Error: {total_error:.4f}")

    def predict(self, X):
        outputs = []
        for x in X:
            net = self.net_input(x)
            outputs.append(1 if net >= 0 else -1)
        return np.array(outputs)


# -----------------------------------------
# Bipolar Inputs for Logic Gates
# -----------------------------------------
X = np.array([
    [-1, -1],
    [-1, +1],
    [+1, -1],
    [+1, +1]
])

# -------- OR Gate (bipolar) ----------
T = np.array([-1, +1, +1, +1])

# Other gates:
# AND  = [-1, -1, -1, +1]
# NAND = [+1, +1, +1, -1]
# NOR  = [+1, -1, -1, -1]

# -----------------------------------------

adaline = AdalineBipolar(input_size=2, learning_rate=0.1, epochs=20)
adaline.train(X, T)

print("\nFinal Weights:", adaline.weights)
print("Final Bias:", adaline.bias)

pred = adaline.predict(X)
print("Predictions:", pred)
print("Targets:", T)
