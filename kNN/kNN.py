import numpy as np
import sys, os


class kNN:
    def __init__(self, k, xTrain, yTrain):
        self.k = k
        self.data = {}
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.distanceType = "Euclidean"
        return

    def setDistanceMetric(self, distance):
        if distance == "Euclidean":
            self.distanceType = "Euclidean"
        elif distance == "Manhattan":
            self.distanceType = "Manhattan"
        elif distance == "Hamming":
            self.distanceType = "Hamming"
        else:
            print("Unsupported distance type. Not changing existing type")
        return

    def hammingDistance(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError("Input strings must have the same length")

        return np.sum(x1 != x2)


    def euclideanDistance(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError("Input strings must have the same length")

        return np.linalg.norm(x1 - x2)

    def manhattanDistance(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError("Input strings must have the same length")

        return np.sum(np.abs(x1 - x2))

    def distance(self, x1, x2):
        if self.distanceType == "Euclidean":
            return self.euclideanDistance(x1, x2)
        elif self.distanceType == "Manhattan":
            return self.manhattanDistance(x1, x2)
        else:
            return self.hammingDistance(x1, x2)
    
    def predict(self, inputVector):
        kNN = []
        for vector in self.xTrain:
            kNN.append(self.distance(inputVector, vector))

        kIndices = np.argsort(kNN)[:self.k]

        kLabels = [self.yTrain[i] for i in kIndices]

        return np.sum(kLabels)
