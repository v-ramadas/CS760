from DecisionTree import *
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas

def P6(inputFile):
    df = pandas.read_csv(inputFile, sep = " ", header=None)
    df.columns = ["0", "1", "label"]

    decisionTree = DecisionTree()
    decisionTree.root = decisionTree.makeSubtree(df)

    if decisionTree.root is not None:
        decisionTree.root.print()
        print(decisionTree.root.getNodeCount())

    x = df["0"].values.tolist()
    y = df["1"].values.tolist()
    labels = df["label"].values.tolist()
    count = 0
    for i in range(df.shape[0]):
        if labels[i] != decisionTree.predict([x[i], y[i]]):
            count += 1

    print("Accuracy = ", 1 - count/df.shape[0])

    lab0 = df[df["label"] == 0]
    x0 = lab0["0"].values.tolist()
    y0 = lab0["1"].values.tolist()
    lab1 = df[df["label"] == 1]
    x1 = lab1["0"].values.tolist()
    y1 = lab1["1"].values.tolist()
    plt.scatter(x0, y0, c = "blue")
    plt.scatter(x1, y1, c = "red")
    plt.savefig(sys.argv[1]+'_scatter.pdf', format="pdf")

    #decision boundary scatter plot
    x = np.random.uniform(0.0, 1.0, 10000)
    y = np.random.uniform(0.0, 1.0, 10000)
    pred = []
    for i in range (len(x)):
        tmp = []
        tmp.append(x[i])
        tmp.append(y[i])
        yp = decisionTree.predict([x[i], y[i]])
        pred.append(yp)
    
    df = pandas.DataFrame(list(zip(x, y, pred)),
                   columns =['0', '1', 'label'])
    lab0 = df[df["label"] == 0]
    x0 = lab0["0"].values.tolist()
    y0 = lab0["1"].values.tolist()
    lab1 = df[df["label"] == 1]
    x1 = lab1["0"].values.tolist()
    y1 = lab1["1"].values.tolist()
    plt.scatter(x0, y0, c = "blue")
    plt.scatter(x1, y1, c = "red")
    plt.savefig(sys.argv[1]+'_decision_boundary.pdf', format="pdf")


def P7():
    df = pandas.read_csv("Dbig.txt", sep = " ", header=None)
    df.columns = ["0", "1", "label"]
    df_train = df.sample(n=8192, replace=False)
    df_test = df.drop(df_train.index)

    print(df_train.shape[0])
    print(df_test.shape[0])
    sample_size = [32, 128, 512, 2048, 8192]
    n_list = []
    err_list = []
    for size in sample_size:
        df_iter = df_train.head(size)
        decisionTree = DecisionTree()
        decisionTree.root = decisionTree.makeSubtree(df_iter)

        x_test = df_test["0"].values.tolist()
        y_test = df_test["1"].values.tolist()
        labels = df_test["label"].values.tolist()

        count = 0
        for i in range(len(x_test)):
            y_pred = decisionTree.predict([x_test[i], y_test[i]])
            if y_pred != labels[i]:
                count = count + 1
        total_size = len(x_test)

        error = count/total_size
        print(size, decisionTree.root.getNodeCount(), error)
        n_list.append(size)
        err_list.append(error)

        x = df["0"].values.tolist()
        y = df["1"].values.tolist()
        pred = []
        for i in range (len(x)):
            yp = decisionTree.predict([x[i], y[i]])
            pred.append(yp)
        
        df = pandas.DataFrame(list(zip(x, y, pred)),
                       columns =['0', '1', 'label'])
        lab0 = df[df["label"] == 0]
        x0 = lab0["0"].values.tolist()
        y0 = lab0["1"].values.tolist()
        lab1 = df[df["label"] == 1]
        x1 = lab1["0"].values.tolist()
        y1 = lab1["1"].values.tolist()
        plt.scatter(x0, y0, c = "blue")
        plt.scatter(x1, y1, c = "red")
        plt.savefig('Dbig_decision_boundary_n_' + str(size) + '.pdf', format="pdf")

    plt.close('all')
    plt.rcdefaults()
    plt.plot(n_list, err_list)
    plt.xlabel('Size of training set')
    plt.ylabel('Test error')
    plt.savefig('P7_learning_rate.pdf', format='pdf')


P6(sys.argv[1])
P7()
