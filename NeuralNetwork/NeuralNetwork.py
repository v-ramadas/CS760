import os
import torch
from torch import nn
from torch.utils.data import DataLoader

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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearSigmoidStack(x)
        return logits

    def train(self, dataLoader, lossFunction, optimizer, dataSize, batchSize):
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
            if i >= dataSize - 1:
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
