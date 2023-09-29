from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas
import sys



df = pandas.read_csv(sys.argv[1], sep=" ", header=None)
df.columns = ["0", "1", "label"]

dataset_sizes = [32, 128, 512, 2048, 8192]

error_list = []

n_list = []

for size in dataset_sizes:
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = size)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    error = 1 - classifier.score(X_test, y_test)
    num_nodes = classifier.tree_. node_count
    print("n: ", size, " nodes: ", num_nodes, " error:", error)
    error_list.append(error)
    n_list.append(size)
    plt.plot(n_list, error_list)
    plt.xlabel('Size of training set')
    plt.ylabel('Test error')
    plt.savefig('scikit_learning_rate.pdf', format='pdf')
