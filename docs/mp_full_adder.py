import numpy as np

def act(x): return 1 if x>=0 else 0

def mp_neuron(inputs,weights,threshold):
    total=np.dot(inputs,weights)
    return act(total-threshold)

def half_adder_sum(A,B):
    n1= mp_neuron(np.array([A,1-B]),np.array([1,1]),2)
    n2= mp_neuron(np.array([1-A,B]),np.array([1,1]),2)
    return 1 if n1 or n2 else 0

def full_adder_carry(A,B,C):
    return mp_neuron(np.array([A,B,C]),np.array([1,1,1]),2)


def AND (A,B):
    return mp_neuron(np.array([A,B]),np.array([1,1]),2)
def full_adder_sum(A,B,C):
    s1=half_adder_sum(A,B)
    return half_adder_sum(s1,C)

print("FULL ADDER TABLE")
for a in [0,1]:
    for b in [0,1]:
        for c in [0,1]:
            print(f"{a} {b} {c}= {full_adder_sum(a,b,c)}, {full_adder_carry(a,b,c)}")