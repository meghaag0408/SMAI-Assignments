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

name_file='arcene_train.data'
lables_name='arcene_train.labels'

with open(name_file,'rb') as file_name:
	datal=list(csv.reader(file_name,delimiter=' '))
with open(lables_name,'rb') as f:
	labels=list(csv.reader(f))

labels=np.array(labels)
labels=np.ravel(labels).astype(int)
for i in range(0,len(datal)):
	del datal[i][-1]
	for x in range(0,len(datal[i])):
		datal[i][x]=float(datal[i][x])

print "Principal Componenets Analysis : Components 10"
pca_output = PCA(n_components=10).fit(datal)
dataset_pca_out=pca_output.transform(datal)

print "*****SVM with Linear Kernels*********"
clf_op=svm.SVC(kernel='linear')

count = 1
print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[:60],labels[:60])#Teseting
result=clf_op.predict(dataset_pca_out[80:])	#Prediction
conf=confusion_matrix(labels[80:], result, labels=[1, -1])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[80:], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[80:], result)
print "Precision Score:", prec
recall=recall_score(labels[80:], result)
print "Recall Score:", recall
count+=1
print "------------------------"


print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[:60],labels[:60])#Teseting
result=clf_op.predict(dataset_pca_out[0:20])	#Prediction
conf=confusion_matrix(labels[0:20], result, labels=[1, -1])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[0:20], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[0:20], result)
print "Precision Score:", prec
recall=recall_score(labels[0:20], result)
print "Recall Score:", recall
count+=1
print "------------------------"



print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[40:],labels[40:])#Teseting
result=clf_op.predict(dataset_pca_out[20:40])	#Prediction
conf=confusion_matrix(labels[20:40], result, labels=[1, -1])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[20:40], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[20:40], result)
print "Precision Score:", prec
recall=recall_score(labels[20:40], result)
print "Recall Score:", recall
count+=1
print "------------------------"



print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[40:],labels[40:])#Teseting
result=clf_op.predict(dataset_pca_out[40:60])	#Prediction
conf=confusion_matrix(labels[40:60], result, labels=[1, -1])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[40:60], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[40:60], result)
print "Precision Score:", prec
recall=recall_score(labels[40:60], result)
print "Recall Score:", recall
count+=1
print "------------------------"



print
print "------Fold No------", count
clf_op.fit(dataset_pca_out[30:90],labels[30:90])#Teseting
result=clf_op.predict(dataset_pca_out[60:80])	#Prediction
conf=confusion_matrix(labels[60:80], result, labels=[1, -1])
print "Confusion Matrix: "
print conf
acc=accuracy_score(labels[60:80], result)

print "Accuracy: ", acc*100, "%"

prec=precision_score(labels[60:80], result)
print "Precision Score:", prec
recall=recall_score(labels[60:80], result)
print "Recall Score:", recall
count+=1
print "------------------------"








