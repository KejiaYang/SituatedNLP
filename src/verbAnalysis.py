import pickle

import operator

import matplotlib.pyplot as plt


def read_data(filename):
    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)
    
    return data


def countCate(doc, score):
    wordCate = {}
    numCate = {}
    wordScore = {}
    numScore = {}
    i = 0
    for d in doc:
        for w in d:
            for at in w[1]:
                if (at[0] in wordCate):
                    wordCate[at[0]] += 1
                    wordScore[at[0]] += score[i]
                else:
                    wordCate[at[0]] = 1
                    wordScore[at[0]] = score[i]
                if (at[1] in numCate):
                    numCate[at[1]] += 1
                    numScore[at[1]] += score[i]
                else:
                    numCate[at[1]] = 1
                    numScore[at[1]] = score[i]
        i += 1
        
        
    for k, v in wordScore.items():
        wordScore[k] = v / wordCate[k]
        
    for k, v in numScore.items():
        numScore[k] = v / numCate[k]
    
#     print(wordScore)
#     print(numScore)

    return wordCate, numCate, wordScore, numScore


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
print(doc)
# print(doc)
score = read_data('meteor.pkl')
wordCate, numCate, wordScore, numScore = countCate(doc, score)
# plotDiagReverse(wordCate)
# plotDiagReverse(numCate)
# plotDiag(wordCate)
# plotDiag(numCate)


# plotDiagReverse(numScore)
# plotDiag(numScore)
# plotDiagReverse(wordScore)
# plotDiag(wordScore)