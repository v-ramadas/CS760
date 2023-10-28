import math
import numpy as np
import os

characterSet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
smoothingParam = 0.5

def getFrequency(langClass, startNum, endNum):
    charFrequency = {}
    for i in range(startNum, endNum):
        inFile = open(os.path.join('..','languageID', langClass + str(i) + '.txt'), 'r')
        print('Opening', os.path.join('..','languageID', langClass + str(i) + '.txt'))
        fileContents = inFile.read()
        for char in characterSet:
            if charFrequency.get(char) != None:
                charFrequency[char] += fileContents.count(char)
            else:
                charFrequency[char] = fileContents.count(char)

    return charFrequency


p_e = 10.5/(30+3*0.5)
p_s = 10.5/(30+3*0.5)
p_j = 10.5/(30+3*0.5)
p_e = math.log(p_e)
p_s = math.log(p_s)
p_j = math.log(p_j)

dictE = getFrequency('e', 0, 10)
totalE = sum(dictE.values())

dictS = getFrequency('s', 0, 10)
totalS = sum(dictS.values())

dictJ = getFrequency('j', 0, 10)
totalJ = sum(dictJ.values())

print("Likelihood(y=e) : ")
for char in characterSet:
    print(char, (dictE[char]+smoothingParam)/(totalE+len(characterSet)*smoothingParam))
    dictE[char] = (dictE[char]+smoothingParam)/(totalE+len(characterSet)*smoothingParam)

print("likelihood(y=s) : ")
for char in characterSet:
    print(char, (dictS[char]+smoothingParam)/(totalS+len(characterSet)*smoothingParam))
    dictS[char] = (dictS[char]+smoothingParam)/(totalS+len(characterSet)*smoothingParam)

print("likelihood(y=j) : ")
for char in characterSet:
    print(char, (dictJ[char]+smoothingParam)/(totalJ+len(characterSet)*smoothingParam))
    dictJ[char] = (dictJ[char]+smoothingParam)/(totalJ+len(characterSet)*smoothingParam)

for languageClass in ['e', 's', 'j']:
    for i in range(10, 20):
        dictPred = getFrequency(languageClass, i, i+1)

        pXGivenYE = 0
        pXGivenYS = 0
        pXGivenYJ = 0
        for char in characterSet:
            pXGivenYE += dictPred[char]*math.log(dictE[char])
            pXGivenYS += dictPred[char]*math.log(dictS[char])
            pXGivenYJ += dictPred[char]*math.log(dictJ[char])

        print(pXGivenYE, pXGivenYS, pXGivenYJ)
        pEGivenX = pXGivenYE + p_e
        pSGivenX = pXGivenYS + p_s
        pJGivexX = pXGivenYJ + p_j
        print(pEGivenX, pSGivenX, pJGivexX)
        class_ = np.argmax([pEGivenX, pSGivenX, pJGivexX])
        print(languageClass, " Prediction = ", class_)
