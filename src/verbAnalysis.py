import pickle

import operator

import matplotlib.pyplot as plt


def read_data(filename):
    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)
    
    return data


def countCate(doc):
    wordCate = {}
    numCate = {}
    for d in doc:
        for w in d:
            for at in w[1]:
                if (at[0] in wordCate):
                    wordCate[at[0]] += 1
                else:
                    wordCate[at[0]] = 1
                if (at[1] in numCate):
                    numCate[at[1]] += 1
                else:
                    numCate[at[1]] = 1
                
    return wordCate, numCate


def plotDiagReverse(dictionary):
    left = []
    height = []
    label = []
    i = 1
    sorted_d = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_d)
    for ele in sorted_d:
        left.append(i)
        label.append(ele[0])
        height.append(ele[1])
        if (i == 10):
            break
        i += 1
        
    plt.bar(left, height, tick_label = label, width = 0.8, color = ['red', 'green'])
    plt.xticks(rotation=90)
    

def plotDiag(dictionary):
    left = []
    height = []
    label = []
    i = 1
    sorted_d = sorted(dictionary.items(), key=operator.itemgetter(1))
    print(sorted_d)
    for ele in sorted_d:
        left.append(i)
        label.append(ele[0])
        height.append(ele[1])
        if (i == 10):
            break
        i += 1
        
    plt.bar(left, height, tick_label = label, width = 0.8, color = ['red', 'green'])
    plt.xticks(rotation=90)


doc = read_data('verbNet.pkl')
wordCate, numCate = countCate(doc)
plotDiagReverse(wordCate)
plotDiagReverse(numCate)
plotDiag(wordCate)
plotDiag(numCate)