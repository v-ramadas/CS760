import sys, os

from kNN.kNN import *
from LogisticRegression.LogisticRegression import *
from sklearn import metrics
import warnings
import pandas
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
df = pandas.read_csv("kNN/emails.csv")
df['Bias'] = 1

alpha = 0.5 #0.05 and 0.1 are also good
k = 5

xTrain = df.drop(columns=["Email No.","Prediction"])
xTrainkNN = xTrain.drop(columns=["Bias"])
yTrain = df["Prediction"]

xTest = xTrain.iloc[4000:]
xTestkNN = xTrainkNN.iloc[4000:]
yTest = yTrain.iloc[4000:]

alpha = 0.5 #0.05 and 0.1 are also good

model1 = LogisticRegression(alpha, xTrain.shape[1])
model1.train(xTrain.iloc[:4000].values, yTrain[:4000].values)
ldPred = model1.predict(xTest.values)

print("Done with LD")

model2 = kNN(k, xTrainkNN.iloc[:4000].values, yTrain[:4000].values)
xVector = xTestkNN.values
yVector = yTest.values

kNNPred = np.array([])

for i in range(len(xVector)):
    y = model2.predict(xVector[i])/k
    kNNPred = np.append(kNNPred, y)
print (yTest.shape)
print (len(kNNPred))
print("Done with kNN")
fpr1,tpr1,thresholds1 = metrics.roc_curve(yTest.values, ldPred)
auc1 = metrics.roc_auc_score(yTest.values, ldPred)
fpr2,tpr2,thresholds2 = metrics.roc_curve(yTest.values, kNNPred)
print("Done with ROC curve")
auc2 = metrics.roc_auc_score(yTest.values, kNNPred)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, color='red', lw=2, label=f'Logistic Regression (AUC = {auc1:.2f})')
plt.plot(fpr2, tpr2, color='blue', lw=2, label=f'kNeighbors Classifier (AUC = {auc2:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
