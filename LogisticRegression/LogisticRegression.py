import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, alpha, numFeatures):
        self.alpha = alpha
        self.weights = [0.1]*numFeatures
        self.numFeatures = numFeatures
        return

    def gradient(self, x, y, yPred):
        gradient = (1.0/self.numFeatures) * np.dot((yPred - y), x)
        return gradient

    def predict(self, x):
        h = np.dot(self.weights, np.transpose(x))
        return sigmoid(h)

    def updateWeights(self, gradient):
        self.weights = self.weights - self.alpha*gradient
        return

    def train(self, x, y):
        count = 0
        while (1):
            yPred = self.predict(x)
            gradient = self.gradient(x, y, yPred)
            self.updateWeights(gradient)
            count = count + 1
            if count > 5000:
                break
        return
