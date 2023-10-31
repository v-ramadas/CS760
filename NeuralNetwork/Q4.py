from NeuralNetwork import *
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import math


learningRate = 0.01
batchSize = 32

mnistDataTrain = datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor())
trainDataLoader = torch.utils.data.DataLoader(mnistDataTrain, batch_size=batchSize, shuffle=False)
mnistDataTest = datasets.MNIST('.', train=False, download=True, transform=transforms.ToTensor())
testDataLoader = torch.utils.data.DataLoader(mnistDataTest, batch_size=1, shuffle=False)

model = NeuralNetworkTorch()
model.initWeights("zeros")
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

misPred = []
epochList = []
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model.train(trainDataLoader, lossFunction, optimizer, batchSize)
    correct = model.test(testDataLoader, lossFunction)
    misPred.append(correct)
    epochList.append(t)
print("Test Error", correct)

plt.plot(epochList, misPred)
plt.grid()
#plt.savefig('pytorch_nn_learning_curve.pdf', format='pdf')
plt.show()
