import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import csv

train_factor = 0.02
init_weight = 0.5
epochCount = 10
    
def buildNetwork(layerSizes, X):
    # Generate the weight matrix W for each layer
    # It will be stored as a python array of 2D matrices
    W = []
    for i in range(len(layerSizes) - 1):
        # print("{0} {1}".format(layerSizes[i], layerSizes[i+1]))
        w = np.random.random((layerSizes[i], layerSizes[i+1]) ) * init_weight - init_weight/2
        W.append(w)
    
    # print(W)
    # Generate the bias values
    # Stored as list of bias arrays
    b = []
    for i in range(len(layerSizes) - 1):
        b.append(np.random.random((1, layerSizes[i+1])) * init_weight - init_weight/2)
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
    
def setNetwork(layerSizes, model, biasFile, weightsFile):
    W = model['W'] # Weights
    b = model['b'] # Biases
    L = model['L'] # Layers
    
    with open(biasFile, 'r') as csvfile:
        for i in range(len(layerSizes) - 1):
            row = next(csvfile).strip().strip("'")
            numbers = iter([x.strip() for x in row.split(',')])
            next(numbers)
            for j in range(layerSizes[i+1]):
                b[i][0][j] = float(next(numbers))
                
    with open(weightsFile, 'r') as csvfile:
        for i in range(len(layerSizes) - 1):
            rows = layerSizes[i]
            cols = layerSizes[i+1]
            for r in range(rows):
                row = next(csvfile).strip().strip("'")
                numbers = iter([x.strip() for x in row.split(',')])
                next(numbers)
                for c in range(cols):
                    W[i][r][c] = float(next(numbers))

def outputNetwork(layerSizes, model, biasFile, weightsFile):
    W = model['W'] # Weights
    b = model['b'] # Biases
    L = model['L'] # Layers
        
    with open(biasFile, 'w') as csvfile:
        for i in range(len(layerSizes) - 1):
            strs = ["%.10f" % number for number in b[i][0]]
            csvfile.write(','.join(strs) + '\n')
            
    with open(weightsFile, 'w') as csvfile:
        for i in range(len(layerSizes) - 1):
            for r in range(W[i].shape[0]):
                strs = ["%.10f" % number for number in W[i][r]]
                csvfile.write(','.join(strs) + '\n')

def softmax(x):
    shiftx = x - np.max(x)
    exp_scores = np.exp(shiftx)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def activation(x):
    return np.maximum(0, x) # ReLU

def predict(model, X):
    yHat = forwardProp(model, X)
    predict = probToPrediction(yHat)
    return predict

def probToPrediction(yHat):
    return np.argmax(yHat, axis=1)

def forwardProp(model, inputLayer):
    layerCount = len(model['layers'])
    W = model['W'] # Weights
    b = model['b'] # Biases
    L = model['L'] # Layers
    L[0] = inputLayer
    for i in range(layerCount-1):
        L[i+1] = L[i].dot(W[i]) + b[i]
        if i == layerCount - 2:
            L[i+1] = softmax(L[i+1])
        else:
            L[i+1] = activation(L[i+1])
    
    return L[layerCount-1]

def shuffleAndSplit(X, Y):
    con = np.concatenate((X, Y), axis=1)
    np.random.shuffle(con)
    interval = int(con.shape[0] / 5)
    cons = [
        con[0:interval],
        con[interval:interval*2],
        con[interval*2:interval*3],
        con[interval*3:interval*4],
        con[interval*4:]
    ]
    batches = []
    for c in cons:
        x = c[:, :X.shape[1]]
        y = c[:, X.shape[1]:]
        batches.append((x,y))
    return batches


def backProp(model, yHat, Y):
    layerCount = len(model['layers'])
    W = model['W'] # Weights
    b = model['b'] # Biases
    L = model['L'] # Layers
    
    deltas = [0] * layerCount
    dWs = [0] * layerCount
    dbs = [0] * layerCount
    
    for i in reversed(range(layerCount-1)):
        if i == layerCount - 2:
            deltas[i] = -(Y - yHat)
            deltas[i] /= deltas[i].shape[0]
        else:
            deltas[i] = np.dot(deltas[i+1], W[i+1].T)
            
        dWs[i] = np.dot(L[i].T, deltas[i])
        dbs[i] = np.sum(deltas[i], axis = 0)
        
    for i in range(layerCount-1):
        W[i] -= train_factor * dWs[i]
        b[i] -= train_factor * dbs[i]


def costFunction(yHat, Y):
    # Cross entropy cost function
    totalLoss = np.sum(-np.multiply(Y, np.log(yHat))) # axis = 1 for individual costs
    return totalLoss / Y.shape[0]

def onehot(y):
    Y = np.zeros((y.shape[0], 4)) 
    for i, yelem in enumerate(np.nditer(y)):
        Y[i,yelem] = 1
    return Y

def trainOnce(layerSizes, filename):
    X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    y = np.array([[0, 0, 0, 1]])
    
    model = buildNetwork(layerSizes, X)
    setNetwork(layerSizes, model, 'Question2_4/c/b-{0}-4.csv'.format(filename), 'Question2_4/c/w-{0}-4.csv'.format(filename))
    
    yHat = forwardProp(model, X)
    backProp(model, yHat, y)
    
    outputNetwork(layerSizes, model, 'Question2_4/c/db-{0}-4.csv'.format(filename), 'Question2_4/c/dw-{0}-4.csv'.format(filename))

def train(X,oY,Y,testX, oTestY, testY, layerSizes, filename):

    model = buildNetwork(layerSizes, X[0:,:])
    trainErrorPlot = np.zeros((epochCount,))
    testErrorPlot = np.zeros((epochCount,))
    trainAccuPlot = np.zeros((epochCount,))
    testAccuPlot = np.zeros((epochCount,))
    for i in range(epochCount):
        for batch in shuffleAndSplit(X,Y):
            x = batch[0]
            y = batch[1]
            
            yHat = forwardProp(model, x[0:,:])
            backProp(model, yHat, y[0:])
            
        yHat = forwardProp(model, X[0:,:])
        error = costFunction(yHat, Y[0:])
        trainErrorPlot[i] = error
        prediction = probToPrediction(yHat)
        trainAccu = np.sum(prediction == oY[0:]) / prediction.shape[0]
        trainAccuPlot[i] = trainAccu
        # yHat = forwardProp(model, X[0:,:])
        # # print(yHat)
        # error = costFunction(yHat, Y[0:])
        # trainErrorPlot[i] = error
        # backProp(model, yHat, Y[0:])
        
        testYHat = forwardProp(model, testX[0:,:])
        testError = costFunction(testYHat, testY[0:])
        testErrorPlot[i] = testError
        testPrediction = probToPrediction(testYHat)
        testAccu = np.sum(testPrediction == oTestY[0:]) / testPrediction.shape[0]
        testAccuPlot[i] = testAccu
                
        print("{0}\tTrain: {1:.4f}({2:.4f})\tTest: {3:.4f}({4:.4f})".format(i, error, trainAccu, testError, testAccu))

        epoches = np.arange(epochCount)

    fig = plt.figure()
    line1, = plt.plot(epoches, trainErrorPlot, label='Training Error')
    line2, = plt.plot(epoches, testErrorPlot, label='Testing Error')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    fig.savefig(filename + '.error.png', dpi=fig.dpi)
    plt.close()
    
    fig = plt.figure()
    line1, = plt.plot(epoches, trainAccuPlot, label='Training Accuracy')
    line2, = plt.plot(epoches, testAccuPlot, label='Testing Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    fig.savefig(filename + '.accu.png', dpi=fig.dpi)
    plt.close()
    
# trainOnce([14, 100, 40, 4], '100-40')
# trainOnce([14]+ [28] * 6 + [4], '28-6')
# trainOnce([14]+ [14] * 28 + [4], '14-28')

X = np.loadtxt(open("Question2_123/x_train.csv", "rb"), delimiter=",")
y = np.loadtxt(open("Question2_123/y_train.csv", "rb"), dtype=np.int32)

testX = np.loadtxt(open("Question2_123/x_test.csv", "rb"), delimiter=",")
testy = np.loadtxt(open("Question2_123/y_test.csv", "rb"), dtype=np.int32)

# Convert into one hot array
Y = onehot(y)
testY = onehot(testy)

train_factor = 0.05
init_weight = 0.5
epochCount = 4000

layerSizes = [X.shape[1], 100, 40, 4]
name = '100-4'
train(X,y,Y, testX, testy, testY, layerSizes, "{0}.{1}.{2}".format(name, train_factor, init_weight))
# layerSizes = [X.shape[1]] + [28] * 6 + [4]
# name = '28^6'
# train(X,y,Y, testX, testy, testY, layerSizes, "{0}.{1}.{2}".format(name, train_factor, init_weight))
# layerSizes = [X.shape[1]] + [14] * 28 + [4]
# name = '14^28'
# train(X,y,Y, testX, testy, testY, layerSizes, "{0}.{1}.{2}".format(name, train_factor, init_weight))

train_factor = 0.001
init_weight = 0.5
epochCount = 10000

layerSizes = [X.shape[1], 100, 40, 4]
name = '100-4'
train(X,y,Y, testX, testy, testY, layerSizes, "{0}.{1}.{2}".format(name, train_factor, init_weight))
# layerSizes = [X.shape[1]] + [28] * 6 + [4]
# name = '28^6'
# train(X,y,Y, testX, testy, testY, layerSizes, "{0}.{1}.{2}".format(name, train_factor, init_weight))
# layerSizes = [X.shape[1]] + [14] * 28 + [4]
# name = '14^28'
# train(X,y,Y, testX, testy, testY, layerSizes, "{0}.{1}.{2}".format(name, train_factor, init_weight))
