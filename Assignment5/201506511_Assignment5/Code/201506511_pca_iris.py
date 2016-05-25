import numpy as np
import matplotlib.pyplot as plt

def read_file_load_dataset(filename, index_of_class):
	f = open(filename, 'rb')
	dataset = f.readlines()
	length = len(dataset)
	sample=[]
	for i in range(0, length):
		test = dataset[i].split(',')
		for j in range(0, len(test)-1):
			test[j] = float(test[j])
		sample.append(test[:-1])

	return dataset, sample

def calculating_pca(sample):
	scatter=np.dot(sample,sample.T)
	eigenvalue, eigenvector=np.linalg.eig(scatter)

	eigenpairs=[]
	for i in range(len(eigenvalue)):
		eigenpairs.append((np.abs(eigenvalue[i]), eigenvector[:,i]))

	pca1 = eigenpairs[0][1].reshape(4,1)
	pca2 = eigenpairs[1][1].reshape(4,1)
	pca2 = eigenpairs[1][1].reshape(4,1)
	pca3 = eigenpairs[2][1].reshape(4,1)
	
	pca_temp1 = np.hstack((pca1, pca2))
	pca_temp2 = np.hstack((pca2, pca3))
	pca_temp3 = np.hstack((pca1, pca3))
	
	pca12=np.dot(pca_temp1.T,sample)	
	pca23=np.dot(pca_temp2.T,sample)
	pca31=np.dot(pca_temp3.T,sample)


	return pca12, pca23, pca31

def save_plot(pca_plot,label_1,label_2,name):

	plt.ylabel(label_1)
	plt.xlabel(label_2)
	class1=plt.plot(pca_plot[0,:50],pca_plot[1,:50],'go',label="Iris-setosa")
	class3=plt.plot(pca_plot[0,100:151],pca_plot[1,100:151],'ro',label="Iris-virginica")
	class2=plt.plot(pca_plot[0,50:100],pca_plot[1,50:100],'bo',label="Iris-versicolor")
	
	location='lower right'
	plt.legend(loc=location)
	plt.savefig(name)
	plt.close()


def plot(pca12, pca23, pca31):
	save_plot(pca12,'PC2','PC1','pca12.png')
	save_plot(pca23,'PC3','PC2','pca23.png')
	save_plot(pca31,'PC3','PC1','pca13.png')
	

if __name__ == '__main__':

	filename = "iris.data"
	index_of_class=4
	dataset, sample = read_file_load_dataset(filename, index_of_class)
	sample = np.array(sample)
	sample = np.transpose(sample)

	feature1,feature2,feature3, feature4   = sample[0], sample[1], sample[2], sample[3]
	mean1, mean2, mean3, mean4 = np.mean(feature1), np.mean(feature2), np.mean(feature3), np.mean(feature4)	

	sample[0] = sample[0] - mean1
	sample[1] = sample[1] - mean2
	sample[2] = sample[2] - mean3
	sample[3] = sample[3] - mean4

	pca12, pca23, pca31 = calculating_pca(sample)
	plot(pca12, pca23, pca31)	