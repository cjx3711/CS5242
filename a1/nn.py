import numpy as np 
import matplotlib.pyplot as plt



def buildNetwork(layerSizes, X):
    # Generate the weight matrix W for each layer
    # It will be stored as a python array of 2D matrices
    W = []
    for i in range(len(layerSizes) - 1):
        print("{0} {1}".format(layerSizes[i], layerSizes[i+1]))
        w = np.random.randn(layerSizes[i], layerSizes[i+1])
        W.append(w)
    
    # print(W)
    # Generate the bias values
    # Stored as a numpy n-1*1 matrix
    b = np.random.randn(len(layerSizes) - 1, 1)
    # print(b)
    
    # Generate the layers
    L = []
    for i in range(len(layerSizes)):
        # Assume 1 training example.
        # Will automatically override when X is put in
        if i == 0:
            l = X
        else:
            l = np.random.randn(X.shape[0], layerSizes[i])
        L.append(l)
    pass

    return {
        'W': W,
        'b': b,
        'L': L,
        'layers': layerSizes
    }

def forwardProp(model, X):
    layerCount = len(model['layers'])
    W = model['W']
    b = model['b']
    L = model['L']
    
    print(L[0].shape)
    print(L[1].shape)
    print(W[0].shape)
    print(b[0].shape)
    L[1] = L[0].dot(W[0]) + b[0]
    L[1] = np.tanh(L[1])
    print(L[1].shape)
    print()
    
    print(L[1].shape)
    print(L[2].shape)
    print(W[1].shape)
    print(b[1].shape)
    L[2] = L[1].dot(W[1]) + b[1]
    L[2] = np.tanh(L[2])
    print(L[2].shape)
    print()
    
    print(L[2].shape)
    print(L[3].shape)
    print(W[2].shape)
    print(b[2].shape)
    L[3] = L[2].dot(W[2]) + b[2]
    exp_scores = np.exp(L[3])
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
    predict = np.argmax(probs, axis=1)
    print(L[3].shape)
    print(probs.shape)
    print(predict.shape)
    print(predict[:10])
    
    
    pass


def backProp():
    pass


X = np.loadtxt(open("Question2_123/x_train.csv", "rb"), delimiter=",")
y = np.loadtxt(open("Question2_123/y_train.csv", "rb"), dtype=np.int32)
# print (X)
# print (y)
Y = np.zeros((X.shape[0], 4))
for i, yelem in enumerate(np.nditer(y)):
    Y[i,yelem] = 1
    
# print(X.shape[0], 4)
# print(Y)

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()


hiddenLayerCount = 1

layerSizes = [X.shape[1], 100, 40, 4]
print(layerSizes)

model = buildNetwork(layerSizes, X)

print(model['L'][-1])

forwardProp(model, X)
