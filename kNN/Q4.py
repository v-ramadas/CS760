import pandas
from kNN import *
import numpy as np
import sys, os
import matplotlib.pyplot as plt

df = pandas.read_csv("emails.csv")

numRows = df.shape[0]
incr = int(numRows/5)
k_list = [1, 3, 5, 7, 10]
acc_list = []
for k in k_list:
    avgAccuracy = 0
    for i in range(5):
        start = i*incr
        end = (i+1) * incr
        xTrain = df.drop(columns=["Email No.","Prediction"])
        xTest = xTrain.iloc[start:end]
        yTrain = df["Prediction"]
        yTest = yTrain.iloc[start:end]
        if start == 0:
            xTrain = xTrain.iloc[end:numRows]
            yTrain = yTrain.iloc[end:numRows]
        elif end == numRows:
            xTrain = xTrain.iloc[0:start]
            yTrain = yTrain.iloc[0:start]
        else:
            xTrain = pandas.concat([xTrain.iloc[0:start], xTrain.iloc[end:numRows]])
            yTrain = pandas.concat([yTrain.iloc[0:start], yTrain.iloc[end:numRows]])
    
        model = kNN(k, xTrain.values, yTrain.values)
    
        xVector = xTest.values
        yVector = yTest.values
        accuracy = 0
        recall = 0
        predictedPos = 0
        for i in range(len(xVector)):
            y = model.predict(xVector[i])/k
            if y > 0.5:
                y = 1
            else:
                y = 0
            if y == yVector[i]:
                accuracy = accuracy + 1
                if y == 1:
                    recall = recall + 1
            if y == 1:
                predictedPos = predictedPos + 1
    
    
        accuracy = accuracy/incr
        precision = recall/predictedPos
        recall = recall/(yVector == 1).sum()
    
        avgAccuracy = avgAccuracy + accuracy
    avgAccuracy = avgAccuracy/5
    acc_list.append(avgAccuracy)

plt.plot(k_list, acc_list)
plt.xlabel('k')
plt.ylabel('Average Accuracy')
plt.title('kNN 5-fold cross validation')
plt.show()
