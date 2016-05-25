import csv
from random import shuffle
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import *
from sklearn import datasets


def loaddata():
	name_file='arcene_train.data'
	lables_name='arcene_train.labels'
	with open(name_file,'rb') as file_name:
		datal=list(csv.reader(file_name,delimiter=' '))
	with open(lables_name,'rb') as f:
		lables=list(csv.reader(f))
	return datal,lables
def svm_call(dataset_pca_out,lables):
	clf=svm.SVC(kernel='linear')
	clf.fit(dataset_pca_out,lables)
	return clf
def cross_validate(clf_in,dataset_pca_out,labels,folds):
	return cross_validation.cross_val_score(clf_in,dataset_pca_out,labels,cv=folds)



def main():
	choice = input("1.iris\n2. arcene\n")
	if choice >2 or choice < 1:
		print "wrong choice"
		exit(0)
	elif choice==1:
		datairis=datasets.load_iris()
		datal=datairis.data
		labels=datairis.target
		comp=2
	else:	
		datal,labels=loaddata()
		labels=np.array(labels)
		labels=np.ravel(labels).astype(int)
		print labels
		length_data=len(datal)
		for i in range(0,length_data):
			del datal[i][-1]
			length_classes=len(datal[i])
			for x in xrange(0,length_classes):
				datal[i][x]=float(datal[i][x])
		comp=10					
	pca_output = PCA(n_components=comp).fit(datal)
	dataset_pca_out=pca_output.transform(datal)
	clf_op=svm_call(dataset_pca_out,labels)	
	result=clf_op.predict(dataset_pca_out)
	print result
	cross_validation_op=cross_validate(clf_op,dataset_pca_out,labels,5)	
	print cross_validation_op

main()



