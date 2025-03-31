# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2024/4/1 10:20

import math

def map(sourcesName,answer,i,ap,p_list):
    # 求map
    curCorrect = 0
    allPrecision = 0
    correctlink = []
    for j in range(0, len(p_list)):
        correctlink = answer.get(sourcesName[i])
        for m in range(0, len(correctlink)):
            if p_list[j][0] == correctlink[m]:
                curCorrect = curCorrect + 1
                curPrecision = curCorrect / (j + 1)
                allPrecision += curPrecision

    if len(correctlink) != 0:
        ap += (allPrecision) / len(correctlink)
    return ap
def precision_recall(answer,sourcesName,recall,i,averagePrecision,p_list):
    # 求固定Recall对应的precision
    precision = []
    correctlink = []
    correctlink = answer.get(sourcesName[i])
    for j in range(0, len(recall)):
        counter = math.ceil(recall[j] * len(correctlink))
        rightLinkNum = counter
        r = 0
        for key in range(0, len(p_list)):
            r = key + 1
            for m in range(0, len(correctlink)):
                if p_list[key][0] == correctlink[m]:
                    counter -= 1
                    break
            if counter == 0:
                break
        precision.append(rightLinkNum / r)
    for w in range(0, 10):
        averagePrecision[w] = averagePrecision[w] + precision[w]
    return averagePrecision
def precision(answer,sourcesName,i,p_list):
    # 求precision
    precision = 0
    if answer.get(sourcesName[i]):
        correctlink = answer.get(sourcesName[i])
        rightLinkNum = len(correctlink)
        counter = rightLinkNum
        r = 0
        for key in range(0, len(p_list)):
            r = key + 1
            for m in range(0, len(correctlink)):
                if p_list[key][0] == correctlink[m]:
                    counter -= 1
                    break
            if counter == 0:
                break
        precision = rightLinkNum / r
    return precision

def recall(answer,sourcesName,i,p_list): # recall
    recall = 0
    if answer.get(sourcesName[i]):
        correctlink = answer.get(sourcesName[i])
        rightLinkNum = len(correctlink)
        r = 0
        for key in range(0, len(correctlink)):
            for m in range(0, len(correctlink)):
                if p_list[key][0] == correctlink[m]:
                    r += 1
        recall = r / rightLinkNum
    return recall

def fn(n,precision,recall):
    return (1+n*n)*recall*precision/(n*n*precision+recall)