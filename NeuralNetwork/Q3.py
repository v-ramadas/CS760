import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math

def assignLabel(label):
    Y = np.zeros(10)
    Y[label] = 1
    return Y

def softmax(A3):
    A3 = np.squeeze(A3)
    softmax = []
    Y_hat = []
    for elem in A3:
        softmax.append(np.exp(elem))
    for elem in softmax:
        Y_hat.append(elem/(np.sum(softmax)))
    Y_hat = np.array(Y_hat)
    return Y_hat

def dSoftmax(Y, Y_hat):
    return (Y_hat-Y)

def sigmoid(Z):
    A = []
    A = 1/(1+np.exp(-Z))
    return A
 
def dSigmoid(A):
    A = A.squeeze()
    A_ = np.diag(A*(1-A))
    return A_

def loss(Y, Y_hat):
    L = 0
    for i,y in enumerate(Y): 
        L -= y*math.log(Y_hat[i])
    return L

def forwardPass(X, W1, W2, W3):
    Z1 = np.matmul(W1.T, X) 
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2.T, A1)
    A2 = sigmoid(Z2)
    Z3 = np.matmul(W3.T, A2)
    Y  = softmax(Z3)
    return Y, A2, A1

def backwardPropogation(Y, Y_hat, A2, A1, W3, W2, W1, X, learningRate):
    dA3 = np.atleast_2d(dSoftmax(Y, Y_hat)).T 
    dW3 = np.matmul(A2, dA3.T) 

    dA2 = np.matmul(W3, dA3) 
    dZ2 = dSigmoid(A2) 
    dZ2 = np.matmul(dZ2, dA2) 
    dW2 = np.matmul(A1, dZ2.T) 

    dA1 = np.matmul(W2, dZ2) 
    dZ1 = dSigmoid(A1) 
    dZ1 = np.matmul(dZ1, dA1) 
    dW1 = np.matmul(X, dZ1.T) 
    
    W3 = W3 - learningRate*dW3
    W2 = W2 - learningRate*dW2
    W1 = W1 - learningRate*dW1
    
    return W3, W2, W1
    

mnist_data_train = torchvision.datasets.MNIST('.', train=True,download=True, transform=ToTensor())
train_data_loader = torch.utils.data.DataLoader(mnist_data_train, batch_size=1, shuffle=False)
mnist_data_test = torchvision.datasets.MNIST('.', train=False,download=True, transform=ToTensor())
mis_pred_list = []
epoch_list = []
dataSize = len(train_data_loader.dataset)
W1 = np.random.uniform(low=-1.0, high=1.0, size=(784, 300))
W2 = np.random.uniform(low=-1.0, high=1.0, size=(300, 200))
W3 = np.random.uniform(low=-1.0, high=1.0, size=(200, 10))
for epoch in range(0, 20):
    for i in range(0, dataSize):
        train_features, train_labels = mnist_data_train[i]
        train_features_flatten = torch.flatten(train_features.squeeze())
        label = train_labels
        X = train_features_flatten.numpy()
        X = np.atleast_2d(X).T
        Y = assignLabel(label)
        Y_hat, A2, A1 = forwardPass(X, W1, W2, W3)
        W3, W2, W1 = backwardPropogation(Y, Y_hat, A2, A1, W3, W2, W1, X, 0.01)

    misPrediction = 0
    for i in range(0, len(mnist_data_test)):
        test_features, test_label = mnist_data_test[i]
        test_features_flatten = torch.flatten(test_features.squeeze())
        X = test_features_flatten.numpy()
        Y_hat, A2, A1 = forwardPass(X, W1, W2, W3)
        pred = np.argmax(Y_hat)
        if test_label != pred:
            misPrediction += 1
    print(dataSize, misPrediction, 1-(misPrediction/len(mnist_data_test)))
    mis_pred_list.append(1-(misPrediction/len(mnist_data_test)))
    epoch_list.append(epoch)
        
print(mis_pred_list)
plt.plot(epoch_list, mis_pred_list)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
