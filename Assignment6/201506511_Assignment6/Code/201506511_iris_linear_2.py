import csv
from random import shuffle
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import *
from sklearn import datasets
from sklearn.metrics import confusion_matrix

datairis=datasets.load_iris()
datal=datairis.data
labels=datairis.target


print "Principal Componenets Analysis : Components 1"
pca_output = PCA(n_components=2).fit(datal)
dataset_pca_out=pca_output.transform(datal)

print "*****SVM with Linear Kernels : IRIS DATA SET*********"
clf_op=svm.SVC(kernel='rbf')

count = 1
print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[:110],labels[:110])#Teseting
result=clf_op.predict(dataset_pca_out[:30])	#Prediction

conf=confusion_matrix(labels[:30], result, labels=[0, 1, 2])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[:30], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[:30], result)
print "Precision Score:", prec
recall=recall_score(labels[:30], result)
print "Recall Score:", recall
count+=1
print "------------------------"


print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[:110],labels[:110])#Teseting
result=clf_op.predict(dataset_pca_out[30:60])	#Prediction

conf=confusion_matrix(labels[30:60], result, labels=[0, 1, 2])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[30:60], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[30:60], result)
print "Precision Score:", prec
recall=recall_score(labels[30:60], result)
print "Recall Score:", recall
count+=1
print "------------------------"

print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[:110],labels[:110])#Teseting
result=clf_op.predict(dataset_pca_out[60:90])	#Prediction

conf=confusion_matrix(labels[60:90], result, labels=[0, 1, 2])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[60:90], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[60:90], result)
print "Precision Score:", prec
recall=recall_score(labels[60:90], result)
print "Recall Score:", recall
count+=1
print "------------------------"

print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[:110],labels[:110])#Teseting
result=clf_op.predict(dataset_pca_out[90:120])	#Prediction

conf=confusion_matrix(labels[90:120], result, labels=[0, 1, 2])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[90:120], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[90:120], result)
print "Precision Score:", prec
recall=recall_score(labels[90:120], result)
print "Recall Score:", recall
count+=1
print "------------------------"

print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[:110],labels[:110])#Teseting
result=clf_op.predict(dataset_pca_out[120:150])	#Prediction

conf=confusion_matrix(labels[120:150], result, labels=[0, 1, 2])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[120:150], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[120:150], result)
print "Precision Score:", prec
recall=recall_score(labels[120:150], result)
print "Recall Score:", recall
count+=1
print "------------------------"








