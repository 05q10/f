import numpy as np

X = np.array([[0, 0, 1, 1],[1, 0, 0, 0],[0, 1, 1, 0],[0, 0, 0, 1]])

W=np.array([ [0.2, 0.9],[0.4, 0.7],[0.6, 0.5],[0.8, 0.3]])


num_clusters=2
alpha=0.25

def eucledian_distance(w,x):
    return np.sum((w - x)**2)

def weight_update(w,x,alpha):
    return w+alpha*(x-w)

#training

for idx, x in enumerate(X, start=1):
    distance=[]
    for j in range(num_clusters):
        d=eucledian_distance(W[:,j],x)
        distance.append(d)
    
    winner=np.argmin(distance)
    print(f"winner : {winner+1}")

    #updates
    W[:,winner]=weight_update(W[:,winner],x,alpha)
    print(W)