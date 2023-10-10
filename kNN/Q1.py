from kNN import *
import pandas
import sys, os
import matplotlib.pyplot as plt
import numpy as np


df = pandas.read_csv("D2z.txt", sep=" ", header=None)
df.columns=["x1", "x2", "label"]

xTrain = df[["x1", "x2"]].values
yTrain = df["label"].values

model = kNN(1, xTrain, yTrain)

x1 = np.arange(0.0, 2.1, 0.1)
x2 = np.arange(-2.0, 0, 0.1)

inputVector = np.concatenate((x1, x2), axis=0)

x = -2
xTest = [[0, 0]]
lenInputVector = len(inputVector)
for i in range(0, lenInputVector):
    xArray = np.full((lenInputVector), x)
    row = np.column_stack((xArray, inputVector))
    xTest = np.concatenate((xTest, row), axis=0)
    x = x+0.1

yTest = []
for x in xTest:
    yTest.append(model.predict(x))

print(df.size)
label0 = df[df["label"] == 0]
x1_0 = label0["x1"].values.tolist()
x2_0 = label0["x2"].values.tolist()
label1 = df[df["label"] == 1]
x1_1 = label1["x1"].values.tolist()
x2_1 = label1["x2"].values.tolist()
#fig,ax = plt.subplots()
plt.scatter(x1_0, x2_0, c = "blue")
plt.scatter(x1_1, x2_1, c = "red")
print (len(x1_1))

df_xTest = pandas.DataFrame(xTest, columns=['x1', 'x2'])
dfTest = pandas.DataFrame(list(zip(df_xTest["x1"].to_list(), df_xTest["x2"].to_list(), yTest)), columns =['x1', 'x2',  'label'])
label0 = dfTest[dfTest["label"] == 0]
x1_0 = label0["x1"].values.tolist()
x2_0 = label0["x2"].values.tolist()
label1 = dfTest[dfTest["label"] == 1]
x1_1 = label1["x1"].values.tolist()
x2_1 = label1["x2"].values.tolist()
#fig,ax = plt.subplots()
plt.scatter(x1_0, x2_0, c = "black", marker='x' )
plt.scatter(x1_1, x2_1, c = "black", marker='+')

plt.show()
print (len(x1_0))

