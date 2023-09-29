import pandas
import sys, os
import math


class Node:
    def __init__(self, feature=None, threshold=None, thenBranch = None, elseBranch = None, label = None):
        self.feature = feature
        self.threshold = threshold
        self.thenBranch = thenBranch
        self.elseBranch = elseBranch
        self.label = label

    def print(self, spacing = 0):
        spacing += 8
        if self.thenBranch is not None:
            self.thenBranch.print(spacing)

        for i in range(spacing):
            print(end = " ")
        print("-> ", self.feature, self.threshold, self.label)

        if self.elseBranch is not None:
            self.elseBranch.print(spacing)

    def getNodeCount(self):
        if (self.thenBranch == None):
            return 1
        l = self.thenBranch.getNodeCount()

        if (self.elseBranch == None):
            return 1
        r = self.elseBranch.getNodeCount()
        
        return 1 + l + r


def entropy(positiveInst, negativeInst):
    if positiveInst == 0 or negativeInst == 0:
        return 0.0
    P_pos = positiveInst/(positiveInst + negativeInst)
    P_neg = 1 - P_pos
    H = -1*(P_pos * math.log2(P_pos) + P_neg*math.log2(P_neg))
    return H


class DecisionTree:

    def __init__(self):
        self.root = None
        self.features = 2

    
    def calculateInfoGain(self, trainInst, feature, candidateSplit, split, h_y):
        #h_y = entropy(split, trainInst[str(feature)].shape[0] - split)
        threshold = trainInst.iloc[split, feature]
        trainInstThen = trainInst[trainInst[str(feature)] >= threshold]
        trainInstElse = trainInst[trainInst[str(feature)] < threshold]

        h_y_then = entropy(trainInstThen[trainInstThen["label"] == 0].shape[0], trainInstThen[trainInstThen["label"] == 1].shape[0])
        h_y_else = entropy(trainInstElse[trainInstElse["label"] == 0].shape[0], trainInstElse[trainInstElse["label"] == 1].shape[0])
        thenNormParam = trainInstThen.shape[0]/(trainInstThen.shape[0] + trainInstElse.shape[0])
        elseNormParam = 1.0 - thenNormParam
        infoGain = h_y - (thenNormParam*h_y_then + elseNormParam*h_y_else)
        h_d = entropy(trainInstThen.shape[0], trainInstElse.shape[0])
        return infoGain, h_d, threshold
    
    def canStopSplit(self, candidateSplit):
        if not any(candidateSplit):
            return True
        return False

    def makeSubtree(self, trainInst):
        candidateSplit = self.determineCandidateSplits(trainInst)

        if self.canStopSplit(candidateSplit):
            leafNode = Node(None, None, None, None, trainInst.iloc[0]["label"])
            return leafNode
        else:
            feature, threshold, label = self.findBestSplit(trainInst, candidateSplit)
            if threshold is None:
                leafNode = Node(None, None, None, None, label)
                return leafNode

            trainInst_sorted = trainInst.sort_values(by=str(feature))
            trainInstThen = trainInst_sorted[trainInst_sorted[str(feature)] >= threshold]
            trainInstElse = trainInst_sorted[trainInst_sorted[str(feature)] < threshold]
            thenSubtree = self.makeSubtree(trainInstThen)
            elseSubtree = self.makeSubtree(trainInstElse)
            node = Node(feature, threshold, thenSubtree, elseSubtree, None)
            return node
    
    def determineCandidateSplits(self, trainInst):
        candidateSplit = []
        if trainInst.empty:
            return candidateSplit
        for column in trainInst.columns[:-1]:
            trainInst_sorted = trainInst.sort_values(by=str(column))
            featureSplit = []
            refLabel = trainInst_sorted.iloc[0]["label"]
            for i in range(trainInst_sorted.shape[0]):
                if refLabel != trainInst_sorted.iloc[i]["label"]:
                    refLabel = trainInst_sorted.iloc[i]["label"]
                    featureSplit.append(i)
            candidateSplit.append(featureSplit)
        return candidateSplit
    
    def findBestSplit(self, trainInst, candidateSplit):
        maxGainRatio = 0
        bestFeature = 0
        bestThreshold = None
        label = 0
        h_y = entropy(trainInst[trainInst.label == 0].shape[0], trainInst[trainInst.label == 1].shape[0])
        if h_y !=  0:
            for feature in range(self.features):
                trainInst_sorted = trainInst.sort_values(by=[str(feature)])
                if len(candidateSplit[feature]) != 0:
                    for split in candidateSplit[feature]:
                        infoGain, h_d, threshold = self.calculateInfoGain(trainInst_sorted, feature, candidateSplit, split, h_y)
                        if h_d == 0:
                            continue
                        gainRatio = infoGain/h_d
                        if (gainRatio > maxGainRatio):
                            maxGainRatio = gainRatio
                            bestFeature = feature
                            bestThreshold = threshold
        if bestThreshold is not None:
            if trainInst_sorted[trainInst_sorted.label == 1].shape[0] > trainInst_sorted[trainInst_sorted.label == 0].shape[0]:
                label = 1
        return bestFeature, bestThreshold, label

    def predict(self, inputVector):
        curNode = self.root
        label = 1
        while(curNode is not None):
            if curNode.feature is not None and curNode.threshold is not None:
                if inputVector[curNode.feature] >= curNode.threshold:
                    curNode = curNode.thenBranch
                else:
                    curNode = curNode.elseBranch
            else:
                label = curNode.label
                curNode = curNode.thenBranch
        return label

def visualize(root, level, parent):
    count = 0
    subTree = None
    if(parent != None):
        if(parent.thenBranch == root):
            subTree = "thenBranch";
        elif(parent.elseBranch == root):
            subTree = "elseBranch";
        else:
            subTree = None;
            
    if root.threshold is None:
        #Visualisation Print Statement
        print("{ node :",id(root), "\n parentNode :", id(parent),  "\n leaf_level :", level, "\n count :", count, "\n predict :", root.label,"\n leaf node }")
        return 1;
    else:
        #Visualisation Print Statement
        print("{ node :",id(root), "\n parentNode :", id(parent), "\n root_level :", level, "\n threshold :", root.threshold, "\n feature :", root.feature, "\n count :", count, "\n subTree :",subTree, "}")
        
        if(root.thenBranch != None):
            count += visualize(root.thenBranch, level + 1, root)
        if(root.elseBranch != None):
            count += visualize(root.elseBranch, level + 1, root)
        return count + 1;



