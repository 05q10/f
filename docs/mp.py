# ---------- EXP 2: MP Neuron Implementation ----------
import numpy as np

# Activation function
def activation(y_in):
    return 1 if y_in >= 0 else 0

# MP neuron model
def mp_neuron(W, X, threshold):
    total = np.dot(W, X)
    return activation(total - threshold)

# AND gate using MP neuron
def AND(A, B):
    return mp_neuron(np.array([A, B]), np.array([1, 1]), threshold=2)

# Test AND gate
print("AND Gate Output:")
print("AND(0,0) =", AND(0, 0))
print("AND(0,1) =", AND(0, 1))
print("AND(1,0) =", AND(1, 0))
print("AND(1,1) =", AND(1, 1))


# Half Adder SUM bit using XOR = n1 OR n2
def half_adder(A, B):
    # neuron-1: A AND (NOT B)
    n1 = mp_neuron(np.array([A, 1 - B]), np.array([1, 1]), threshold=2)

    # neuron-2: (NOT A) AND B
    n2 = mp_neuron(np.array([1 - A, B]), np.array([1, 1]), threshold=2)

    # XOR = n1 OR n2
    return 1 if (n1 or n2) else 0


# Test Half Adder (SUM output)
print("\nHalf Adder SUM Output (XOR):")
print("SUM(0,0) =", half_adder(0, 0))
print("SUM(0,1) =", half_adder(0, 1))
print("SUM(1,0) =", half_adder(1, 0))
print("SUM(1,1) =", half_adder(1, 1))
