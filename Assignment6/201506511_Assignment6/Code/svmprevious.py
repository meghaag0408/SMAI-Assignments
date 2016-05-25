import csv
from random import shuffle
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

with open('arcene_train.data','rb') as f:   
    reader=csv.reader(f,delimiter=' ')
    l=list(reader)

with open('arcene_train.labels','rb') as f:
	labels=[line.strip() for line in f]	    

for i in range(len(l)):
	del l[i][-1]
	for j in range(len(l[i])):
		l[i][j]=float(l[i][j])


pca = PCA(n_components=10)
pca.fit(l)
l = pca.transform(l)

clf = svm.SVC(kernel='linear')
clf.fit(l[:80], labels[:80])  

result=clf.predict(l[80:])


conf=confusion_matrix(labels[80:], result, labels=['1', '-1'])

acc=accuracy_score(labels[80:], result)

#prec=precision_score(labels[80:], result)

#rec=rec_score(labels[80:], result)


print conf
print acc
#print prec, rec





