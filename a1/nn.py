import numpy as np 
import matplotlib.pyplot as plt

X = np.loadtxt(open("Question2_123/x_train.csv", "rb"), delimiter=",")
y = np.loadtxt(open("Question2_123/y_train.csv", "rb"), dtype=np.int32)
print (X)
print (y)
Y = np.zeros((X.shape[0], 4))
for i, yelem in enumerate(np.nditer(y)):
    Y[i,yelem] = 1
    
print(X.shape[0], 4)
print(Y)

plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()


hiddenLayerCount = 1


weights, layers = buildNetwork()

layerSizes = np.array([X.shape[1], 100, 40, 4])
print(layerSizes)


def buildNetwork():
    # Generate the weight matrix W for each layer
    # 
    pass

def forwardProp():
    pass


def backProp():
    pass