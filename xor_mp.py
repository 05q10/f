def mp_neuron(inputs, weights, threshold):
    net = sum(i*w for i, w in zip(inputs, weights))
    output = 1 if net >= threshold else 0
    return output


# XOR implemented using MP neuron network
def xor_mp(x1, x2):

    # Neuron 1: OR gate
    or_out = mp_neuron([x1, x2], [1, 1], 1)

    # Neuron 2: AND gate
    and_out = mp_neuron([x1, x2], [1, 1], 2)

    # Neuron 3: NOT(AND)
    nand_out = mp_neuron([and_out], [-1], -0.5)   # threshold = -0.5 simulates NOT

    # Neuron 4: Final AND = OR • NAND
    xor_out = mp_neuron([or_out, nand_out], [1, 1], 2)

    return xor_out


# Test XOR
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"XOR({x1},{x2}) = {xor_mp(x1,x2)}")
