import numpy as np
def activation(x): 
   return 1 if x >= 0 else 0 
 
def mp_neuron(inputs, weights, threshold): 
   total = np.dot(inputs, weights) 
   return activation(total - threshold) 
 
def AND(A, B): 
   return mp_neuron(np.array([A, B]), np.array([1, 1]), 2) 
 
def half_adder_carry(A, B): 
   return AND(A, B) 
 
def half_adder_sum(A, B): 
   n1 = mp_neuron(np.array([A, 1-B]), np.array([1, 1]), 
2) 
   n2 = mp_neuron(np.array([1-A, B]), np.array([1, 1]), 
2) 
   return 1 if n1 or n2 else 0 
 
# Test Half Adder 
print("Half Adder Truth Table") 
print("A B | Sum Carry") 
for A in [0, 1]: 
   for B in [0, 1]: 
       S = half_adder_sum(A, B) 
       C = half_adder_carry(A, B) 
       print(f"{A} {B} |  {S}    {C}")