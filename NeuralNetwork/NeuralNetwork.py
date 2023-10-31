import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.init as init

class NeuralNetworkTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linearSigmoidStack = nn.Sequential(
                nn.Linear(28*28, 300),
                nn.Sigmoid(),
                nn.Linear(300, 200),
                nn.Sigmoid(),
                nn.Linear(200, 10),
                nn.LogSoftmax(),
        )

    def initWeights(self, initType):

        for layer in self.linearSigmoidStack:
            if isinstance(layer, nn.Linear):
                if initType == "zeros":
                    init.constant_(layer.weight, 0.0) 
#                    if layer.bias is not None:
#                        init.constant_(layer.bias, 0) 
                else:
                    init.uniform_(layer.weight, -1, 1)
                    if layer.bias is not None:
                        init.uniform_(layer.bias, -1, 1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearSigmoidStack(x)
        return logits

    def train(self, dataLoader, lossFunction, optimizer, batchSize):
        size = len(dataLoader.dataset)
        i = 0
        for batch, (X, y) in enumerate(dataLoader):
            pred = self(X)
            loss = lossFunction(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            i += batchSize
            if i > size:
                break
        return


    def test(self, dataLoader, lossFunction):
        size = len(dataLoader.dataset)
        numBatches = len(dataLoader)
        testLoss = 0
        correct = 0

        with torch.no_grad():
            for X, y in dataLoader:
                pred = self(X)
                testLoss += lossFunction(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        testLoss /= numBatches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")
        return correct
