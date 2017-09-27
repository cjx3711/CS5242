import numpy as np 
# import matplotlib.pyplot as plt



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
    # Stored as list of bias arrays
    b = []
    for i in range(len(layerSizes) - 1):
        b.append(np.zeros((1, layerSizes[i+1])))
    # print(b)
    
    # Generate the layers
    L = []
    for i in range(len(layerSizes)):
        # Assume 1 training example.
        # Will automatically override when X is put in
        if i == 0:
            l = X
        else:
            l = np.zeros((X.shape[0], layerSizes[i]))
        L.append(l)
    pass

    return {
        'W': W,
        'b': b,
        'L': L,
        'layers': layerSizes
    }

def softmax(x):
    # print(x)
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def activation(x):
    return np.maximum(0, x) # ReLU

def activationPrime(x):
    return 1

def predict(model, X):
    yHat = forwardProp(model, X)
    predict = probToPrediction(yHat)
    print(predict.shape)
    print(predict[:10])
    return predict

def probToPrediction(yHat):
    return np.argmax(yHat, axis=1)

def forwardProp(model, X):
    layerCount = len(model['layers'])
    W = model['W'] # Weights
    b = model['b'] # Biases
    L = model['L'] # Layers
    
    # print("L[0].shape: {0}".format(L[0].shape))
    # print("L[1].shape: {0}".format(L[1].shape))
    # print("W[0].shape: {0}".format(W[0].shape))
    # print("b[0].shape: {0}".format(b[0].shape))
    L[1] = L[0].dot(W[0]) + b[0]
    L[1] = activation(L[1])
    # print("avg L[1]: {0}".format(np.average(L[1])))
    # print("L[1].shape: {0}".format(L[1].shape))
    # print()
    
    # print("L[1].shape: {0}".format(L[1].shape))
    # print("L[2].shape: {0}".format(L[2].shape))
    # print("W[1].shape: {0}".format(W[1].shape))
    # print("b[1].shape: {0}".format(b[1].shape))
    L[2] = L[1].dot(W[1]) + b[1]
    L[2] = activation(L[2])
    # print("avg L[2]: {0}".format(np.average(L[2])))
    
    # print("L[2].shape: {0}".format(L[2].shape))
    # print()
    
    # print("L[2].shape: {0}".format(L[2].shape))
    # print("L[3].shape: {0}".format(L[3].shape))
    # print("W[2].shape: {0}".format(W[2].shape))
    # print("b[2].shape: {0}".format(b[2].shape))
    L[3] = L[2].dot(W[2]) + b[2]
    # print("avg L[3]: {0}".format(np.average(L[3])))
    yHat = L[3] = softmax(L[3])
    # print("L[3].shape: {0}".format(L[3].shape))
    # print("yHat.shape: {0}".format(yHat.shape))

    return yHat


def backProp(model, yHat, Y):
    layerCount = len(model['layers'])
    W = model['W'] # Weights
    b = model['b'] # Biases
    L = model['L'] # Layers
    
    delta3 = -(Y - yHat)
    delta3 /= delta3.shape[0]
    dW3 = np.dot(L[2].T, delta3)
    db3 = np.sum(delta3, axis = 0)
    
    delta2 = np.dot(delta3, W[2].T)
    dW2 = np.dot(L[1].T, delta2)
    db2 = np.sum(delta2, axis = 0)
    
    delta1 = np.dot(delta2, W[1].T)
    dW1 = np.dot(L[0].T, delta1)
    db1 = np.sum(delta1, axis = 0)
    
    W[0] -= 0.1 * dW1
    W[1] -= 0.1 * dW2
    W[2] -= 0.1 * dW3
    b[0] -= 0.1 * db1
    b[1] -= 0.1 * db2
    b[2] -= 0.1 * db3
    
    # print("delta3.shape: {0}".format(delta3.shape))
    # print("dW3.shape: {0}".format(dW3.shape))
    # print("db3.shape: {0}".format(db3.shape))
    # print("delta2.shape: {0}".format(delta2.shape))
    # print("dW2.shape: {0}".format(dW2.shape))
    # print("db2.shape: {0}".format(db2.shape))
    # print("delta1.shape: {0}".format(delta1.shape))
    # print("dW1.shape: {0}".format(dW1.shape))
    # print("db1.shape: {0}".format(db1.shape))
    # for i in range(3):
    #     print("L[{0}].shape: {1}".format(i, L[i].shape))
    # 
    # for i in range(3):
    #     print("W[{0}].shape: {1}".format(i, W[i].shape))
    #     
    # for i in range(3):
    #     print("b[{0}].shape: {1}".format(i, b[i].shape))


def costFunction(yHat, Y):
    # Cross entropy cost function
    totalLoss = np.sum(-np.multiply(Y, np.log(yHat))) # axis = 1 for individual costs
    return totalLoss / Y.shape[0]

X = np.loadtxt(open("Question2_123/x_train.csv", "rb"), delimiter=",")
y = np.loadtxt(open("Question2_123/y_train.csv", "rb"), dtype=np.int32)
# print (X)
# print (y)
# Convert into one hot array
Y = np.zeros((X.shape[0], 4)) 
for i, yelem in enumerate(np.nditer(y)):
    Y[i,yelem] = 1

# print(X.shape[0], 4)
# print(Y)

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()

layerSizes = [X.shape[1], 100, 40, 4]
# print(layerSizes)

model = buildNetwork(layerSizes, X[0:,:])

# print(model['L'][-1])

for i in range(500):
    yHat = forwardProp(model, X[0:,:])
    # print(model['L'][-1])
    predict = probToPrediction(yHat)
    error = costFunction(yHat, Y[0:])
    backProp(model, yHat, Y[0:])
    print("Error: {0}".format(error))



