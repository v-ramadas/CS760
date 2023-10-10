from LogisticRegression import *
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import warnings

warnings.filterwarnings('ignore')
df = pandas.read_csv("emails.csv")
df['Bias'] = 1


alpha = 0.5 #0.05 and 0.1 are also good

fout = open("Q3_results.txt", "w")

numRows = df.shape[0]
incr = int(numRows/5)
for i in range(5):
    start = i*incr
    end = (i+1) * incr
    xTrain = df.drop(columns=["Email No.","Prediction"])
    yTrain = df["Prediction"]
    xTest = xTrain.iloc[start:end]
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

    model = LogisticRegression(alpha, xTrain.shape[1])
    model.train(xTrain, yTrain)
    y = model.predict(xTest.values)
    #y = [1 if x > 0.5 else 0 for x in y]
    y = (y > 0.5).astype(int)

    TP = np.sum((yTest == 1) & (y == 1))
    TN = np.sum((yTest == 0) & (y == 0))
    FN = np.sum((yTest == 1) & (y == 0))
    FP = np.sum((yTest == 0) & (y == 1))
    recall = TP/(TP + FN)
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)

    print(i, accuracy, precision, recall)
    fout.write(str(i) + ',' + str(accuracy) + ',' +  str(precision) + ',' +  str(recall))
