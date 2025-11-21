#auto associative
import numpy as np

def activation(y):
    return np.where(y > 0, 1, -1)
X=np.array([-1,1,1,1]).reshape(4,1)
W=X@X.T
np.fill_diagonal(W, 0)

def test_network (test_x):
  test_x=np.array(test_x).reshape(4,1)
  yin=W@test_x
  y=activation(yin)
  return yin.flatten(),y.flatten()

tests = {
    "Original Input": [-1, 1, 1, 1],
    "One Missing (case 1)": [0, 1, 1, 1],
    "One Missing (case 2)": [-1, 1, 0, 1],
    "One Mistake": [-1, -1, 1, 1],
    "Two Missing": [0, 1, 0, 1],
    "Two Mistakes": [-1, -1, -1, 1]
}

for name , vector in tests.items():
  yin,y=test_network(vector)
  print(f"{name}:{vector}")
  print(f"Net input : {yin}")
  print(f"output : {y}")