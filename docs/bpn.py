import numpy as np
from tabulate import tabulate

# Sigmoid and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)


# Inputs and target
x1, x2 = 0, 1
t = 1
alpha = 0.25

# Initial weights
v11, v21, b1 = 0.6, -0.1, 0.3
v12, v22, b2 = -0.3, 0.4, 0.5
w1, w2, b0 = 0.4, 0.1, -0.2

headers = ["v11", "v12", "v21", "v22", "v01", "v02", "w1", "w2", "w0"]

for epoch in range(1, 151):
    # Forward pass
    zin1 = b1 + v11 * x1 + v21 * x2
    zin2 = b2 + v12 * x1 + v22 * x2
    z1, z2 = sigmoid(zin1), sigmoid(zin2)

    yin = b0 + w1 * z1 + w2 * z2
    y = sigmoid(yin)

    error = (t - y)
    delta_y = error * sigmoid_derivative(y)

    delta_in1 = delta_y * w1
    delta_in2 = delta_y * w2
    delta1 = delta_in1 * sigmoid_derivative(z1)
    delta2 = delta_in2 * sigmoid_derivative(z2)

    # Weight updates
    dw1 = alpha * delta_y * z1
    dw2 = alpha * delta_y * z2
    db0 = alpha * delta_y

    dv11 = alpha * delta1 * x1
    dv21 = alpha * delta1 * x2
    db1_update = alpha * delta1

    dv12 = alpha * delta2 * x1
    dv22 = alpha * delta2 * x2
    db2_update = alpha * delta2

    v11 += dv11
    v12 += dv12
    v21 += dv21
    v22 += dv22
    b1  += db1_update
    b2  += db2_update

    w1 += dw1
    w2 += dw2
    b0 += db0

    weights = [v11, v12, v21, v22, b1, b2, w1, w2, b0]

    if epoch <= 1 or epoch == 150:
        print(f"\n===== Final Updated Weights After Epoch {epoch} =====")
        print(tabulate([weights], headers=headers, floatfmt=".5f", tablefmt="grid"))
        print(f"Output y = {y:.5f}, Error = {error:.5f}")