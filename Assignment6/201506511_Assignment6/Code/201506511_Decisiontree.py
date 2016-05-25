import random
import operator
from math import *
def split(data, axis, val):
    newData = []
    for i in range(len(data)):
        if data[i][axis] != val:
            pass
        else:
            reducedFeat = data[i][:axis]
            temp = axis
            temp = temp +1
            reducedFeat.extend(data[i][temp:])
            newData.append(reducedFeat)
    return newData

def  lodadata(file_name):
    dataset = open(file_name,"r")
    data = []
    lines = dataset.readlines()
    data_with_normalisation = []
    for i in range(len(lines)):
        data.append(lines[i].split(","))
    temp = []
    for x in range(0,len(data)):
        len_data = len(data[x])
        for y in range(0,len_data):
            if y > 0:
                t = float(data[x][y])
                data[x][y]=t
                temp.append(t)
        data_with_normalisation.append(temp)
        temp = []
    return data_with_normalisation

def majority(classList):
    dictionary_class_count={}
    
    for i in range(len(classList)):
        if classList[i] in dictionary_class_count.keys():
            pass
        else:   
            dictionary_class_count[classList[i]] = 0
        temp = dictionary_class_count[classList[i]]
        temp = temp+1
        dictionary_class_count[classList[i]] = temp
    value = sorted(dictionary_class_count.iteritems(),key=operator.itemgetter(1), reverse=True)[0][0]  
    return value

def entropy(data):
    labels={}
    count=0
    entropy = float(0)
    for i in range(len(data)):
        label=data[i][-1]
        count+=1
        if label in labels.keys():
            pass
        else:
            t = 0
            labels[label] = 0
        t+=1
        labels[label]=labels[label]+ 1
    
    for key in labels:
        length = len(data)
        value = float(labels[key])
        temp = value/length
        entropy=entropy-temp* log(temp,2)
    return entropy  

def choose(data):
    temp = data[0]
    length=len(temp)
    bestInfoGain =  0.0
    features = length - 1
    baseEntropy = entropy(data)   
    bestFeat = -1
    newEntropy = 0.0
    for i in range(features):
        uniqueVals = set([ex[i] for ex in data])
        newData_list = []
        for value in uniqueVals:     
            newData = split(data, i, value)
            length = len(newData)
            length_data = len(data)
            length_data = float(length_data)
            temp = newEntropy
            newEntropy=(length/length_data) * entropy(newData)
            newEntropy=newEntropy+temp
            newData_list = []
        compare = baseEntropy - newEntropy
        if(bestInfoGain<compare):
            info = baseEntropy - newEntropy
            bestFeat = i
            bestInfoGain = info
            
        newEntropy = 0.0
    return bestFeat     

def tree(data,labels):
    classList=[]
    for ex in data:
        classList.append(ex[-1])
    length = len(data[0])
    if length == 1:
        return majority(classList) 
    if classList.count(classList[0]) == len(classList):
        return classList[0]
       
    bestFeat = choose(data)
    theTree = {labels[choose(data)]:{}}
    bestFeatLabel = labels[choose(data)]
    del(labels[bestFeat])
    uniqueVals = set([ex[bestFeat] for ex in data])
    for value in uniqueVals:
        split_value = split(data, bestFeat, value)
        theTree[bestFeatLabel][value] = tree(split_value, labels[:])
    return theTree

def cal_accuracy(test_data,dtree, correct):
    len_test = len(test_data)
    for index in xrange(len_test):
        temp = dtree
        while isinstance(temp,dict):
            key = temp.keys()
            t, temp_key_0 = key[0], key[0]
            temp_key=t-1
            value = test_data[index][t-1]
            temp = temp[temp_key_0]
            if value not in temp:
                ina = temp.keys()[0]
                temp = ina
            else:
                temp = temp[value]
        val = test_data[index][-1]
        if temp != val:
            pass
        else:
            t = correct
            t = t + 1
            correct = t
    ret = correct*100
    ret = float(ret)
    ret = ret/len_test
    return ret


def call_5_fold(data,index_start,index_end,total_accuracy,len_data_cf):
    training_data = []
    for fold in xrange(5):
        training_data=[]
        print
        print "******Fold No ", fold , "******"
        temp = index_end
        index_end+=len_data_cf
        index_start = temp
        test_data=[]
        for i in range(index_start, index_end):
            test_data.append(data[i])
        if index_start > 0:
            length = len(training_data)
            training_data.extend(data[0:index_start])
            temp2 = index_end
        temp2 = index_end
        if len(data) > temp2 :
            length = len(data)
            training_data.extend(data[index_end:length])
            
        dtree=tree(training_data,[1,2,3,4])
        accuracy=cal_accuracy(test_data,dtree,0)
        print dtree
        print 'Accuracy',
        print accuracy
        total_accuracy=total_accuracy+ accuracy
    return total_accuracy 

if __name__ == '__main__':
    data=lodadata("hayes-roth.data")
    random.shuffle(data)
    length = len(data)
    length = length/5
    total_accuracy=call_5_fold(data,0,0,0,length)
    print "Average Accuracy= ",
    total_accuracy = total_accuracy/5
    print total_accuracy
    print