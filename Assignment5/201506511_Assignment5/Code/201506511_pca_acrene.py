import numpy as np
import matplotlib.pyplot as plt
import csv

def read_file_load_dataset(filename):
	f = open(filename, 'rb')
	dataset = f.readlines()
	length = len(dataset)

	sample=[]
	for i in range(0, length):
		test = dataset[i].split(' ')
		for j in range(0, len(test)-1):
			test[j] = float(test[j])
		sample.append(test[:-1])

	return dataset, sample	


def plot(x1, y1, x2, y2, xlabel, ylabel, name):
	plt.ylabel(label_1)
	plt.xlabel(label_2)
	class2=plt.plot(x2,y2,'bo',label="-1")		
	class1=plt.plot(x1,y1,'ro',label="1")
	location='lower right'
	plt.legend(loc=location)
	plt.savefig(name)
	plt.close()


def plot2(x, eigenvalue, xlabel, ylabel, name):
	plt.ylabel(label_1)
	plt.xlabel(label_2)
	class1=plt.plot(x,eigenvalue)
	location='lower right'
	plt.legend(loc=location)
	plt.savefig(name)
	plt.close()


def calculating_eigenpairs(sample):
	scatter=np.dot(sample,sample.T)
	print scatter.shape
	eigenvalue, eigenvector=np.linalg.eig(scatter)
	eigenpairs=[]	

	for i in range(len(eigenvalue)):
		eigenpairs.append((np.abs(eigenvalue[i]), eigenvector[:,i]))

	index=np.argsort(eigenvalue)
	vector1, vector2 =eigenvector[index[-1]], eigenvector[index[-2]]
	vector1 = vector1.reshape(10000,1)
	vector2 = vector2.reshape(10000,1)

	pc12 = np.hstack((vector1, vector2))
	pca=np.dot(pc12.T,sample)

	sorted(eigenvalue,reverse=True)	
	return eigenvalue, pca


if __name__ == '__main__':

	filename = "arcene_train.data"
	dataset, sample = read_file_load_dataset(filename)
	sample = np.transpose(sample)

	with open('arcene_train.labels','rb') as f:
		reader=csv.reader(f)
		labels=list(reader)	
	x1,y1, x2, y2 = [],[],[],[]
	ar = 10000*[0.0]
	avg = np.array(ar)
	for i in range(0, 10000):
		avg[i]=np.mean(sample[i])

	for i in range(0, 10000):
		for j in range(len(sample[i])):
			sample[i][j]=sample[i][j]-avg[j]
	
	eigenvalue, pca = calculating_eigenpairs(sample)

	for i in range(0, 100):
		if labels[i]==['1']:
			x1.append(pca[0][i])
			y1.append(pca[1][i])
	
	for i in range(0, 100):
		if labels[i]==['-1']:		
			x2.append(pca[0][i])
			y2.append(pca[1][i])

	plot(x1, y1, x2, y2, 'PC1', 'PC2', "pca_large_data12.png")
	x=[]
	for i in range(1,10001):
		x.append(i)
	

	plot2(x, eigenvalue, 'factor_number', 'eigenvalue', 'screeplot.png')

	


	

