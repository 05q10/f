import numpy as np

X=np.array([[1,1],[1,0],[0,1],[0,0]])
T=np.array([1,1,1,-1])
w=np.array([0,0])
b=0.0
lr=1
theta=0

print(f"Initial weights {w} ")
def activation(y_in):
    return 1 if y_in >= 0 else -1

for epoch in range (1,4):
    for i in range (len(X)):
        x=X[i]
        t=T[i]

        yin=np.dot(x,w)+b
        y=activation(yin)

        if y!=t:
            #weight update
            w=w+lr*t*x
            b=b+lr*t
            
        print(f"Input: {x}, Target: {t}, y_in={yin:.2f}, Output={y}, Updated Weights={w}, Bias={b}")