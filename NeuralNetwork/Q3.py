import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math

DEBUG = 0
def print_t(X, name):
    if DEBUG:
        print(name, X.shape)
        print(X)

LR = 0.001

def loss(Y, Y_cap):
    L = 0
    for i,y in enumerate(Y): 
        L -= y*math.log(Y_cap[i])
    
    return L

def convert_onehot(label):
    Y = np.zeros(10)
    Y[label] = 1
    return Y

def softmax(A3):
    Y_cap = []
    eA3 = []
    A3 = np.squeeze(A3)
    #print("A3 = ", A3)
    for a3 in A3:
        eA3.append(np.exp(a3))
    for ea3 in eA3:
        Y_cap.append(ea3/(np.sum(eA3)))
    #print(eA3)
    #print(Y_cap)
    Y_cap = np.array(Y_cap)
    return Y_cap

def d_loss_softmax(Y, Y_cap):
    return (Y_cap-Y)

def sigmoid(Z):
    A = []
    A = 1/(1+np.exp(-Z))
    return A
 
def d_sigmoid(A):
    A = A.squeeze()
    A_ = np.diag(A*(1-A))
    return A_

#Y = 1D array
#Y_cap = 1D array
#X  = 784x1
#W1 = 784x300
#W2 = 300x200
#W3 = 200x10
#A2 = 200x1
#A1 = 300x1

def forward(X, W1, W2, W3):
    Z1 = np.matmul(W1.T, X) 
    print_t(Z1, 'Z1')
    A1 = sigmoid(Z1)
    print_t(A1, 'A1')
    Z2 = np.matmul(W2.T, A1)
    print_t(Z2, 'Z2')
    A2 = sigmoid(Z2)
    print_t(A2, 'A2')
    Z3 = np.matmul(W3.T, A2)
    print_t(Z3, 'Z3')
    Y  = softmax(Z3)
    return Y, A2, A1

def backprop(Y, Y_cap, A2, A1, W3, W2, W1, X):
    dA3 = np.atleast_2d(d_loss_softmax(Y, Y_cap)).T #dA3 = 10x1
    dW3 = np.matmul(A2, dA3.T) #200x1 * 1x10 = 200x10

    dA2 = np.matmul(W3, dA3) #200x10 * 10x1 = 200x1
    dZ2 = d_sigmoid(A2) #dZ2 = 200x200
    dZ2 = np.matmul(dZ2, dA2) #200x200 * 200x1 = 200x1
    dW2 = np.matmul(A1, dZ2.T) #300x1 * 1x200 = 300x200

    dA1 = np.matmul(W2, dZ2) #300x200 * 200x1 = 300x1
    dZ1 = d_sigmoid(A1) #dZ1 = 300x300
    dZ1 = np.matmul(dZ1, dA1) #300x300 * 300x1 = 300x1
    dW1 = np.matmul(X, dZ1.T) #784x1 * 1x300 = 784x300
    
    W3 = W3 - LR*dW3
    W2 = W2 - LR*dW2
    W1 = W1 - LR*dW1
    
    return W3, W2, W1
    

mnist_data_train = torchvision.datasets.MNIST('.', train=True,download=True, transform=ToTensor())
train_data_loader = torch.utils.data.DataLoader(mnist_data_train, batch_size=1, shuffle=False)
mnist_data_test = torchvision.datasets.MNIST('.', train=False,download=True, transform=ToTensor())
mis_pred_list = []
dataset_l = [10, 100, 500, 1000, 5000, 10000, 20000]
for dataset in dataset_l:
    #W1 = np.random.uniform(low=-1.0, high=1.0, size=(784, 300))
    #W2 = np.random.uniform(low=-1.0, high=1.0, size=(300, 200))
    #W3 = np.random.uniform(low=-1.0, high=1.0, size=(200, 10))
    W1 = np.zeros((784, 300))
    W2 = np.zeros((300, 200))
    W3 = np.zeros((200, 10))
    for e in range(0, 20):
        for i in range(0, dataset):
            train_features, train_labels = mnist_data_train[i]

            #img = train_features[0].squeeze()
            #plt.imshow(img, cmap="gray")
            #plt.show()
            #print(f"Label: {label}")
            #print(f"Feature batch shape: {train_features.size()}")
            #print(f"Labels batch shape: {train_labels.size()}")
            train_features_flatten = torch.flatten(train_features.squeeze())
            label = train_labels
            #print("label = ", label)
            X = train_features_flatten.numpy()
            X = np.atleast_2d(X).T
            #print ("X shape = ", X.shape)
            Y = convert_onehot(label)
            #print ("Y shape = ", Y.shape)
            Y_cap, A2, A1 = forward(X, W1, W2, W3)
            print_t(X, 'X')
            print_t(W1, 'W1')
            print_t(W2, 'W2')
            print_t(W3, 'W3')
            print_t(A1, 'A1')
            print_t(A2, 'A2')
            print_t(Y_cap, 'Y_cap')
            #print ("Y_cap shape = ", Y_cap.shape)
            #print ("A2 shape = ", A2.shape)
            #print ("A1 shape = ", A1.shape)
            W3, W2, W1 = backprop(Y, Y_cap, A2, A1, W3, W2, W1, X)
        #print("LOSS =" ,loss(Y, Y_cap))

    mis_pred = 0
    for i in range(0, len(mnist_data_test)):
        test_features, test_label = mnist_data_test[i]
        test_features_flatten = torch.flatten(test_features.squeeze())
        X = test_features_flatten.numpy()
        Y_cap, A2, A1 = forward(X, W1, W2, W3)
        pred = np.argmax(Y_cap)
        if test_label != pred:
            mis_pred += 1
    print(dataset, mis_pred, 1-(mis_pred/len(mnist_data_test)))
    mis_pred_list.append(1-(mis_pred/len(mnist_data_test)))
        
plt.plot(dataset_l, mis_pred_list)
plt.xlabel("Data set size")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.grid()
#plt.savefig('nn_learning_curve_zeroes.pdf', format='pdf')
plt.show()
