import numpy as np
import matplotlib.pyplot as plt

def read_file_load_dataset(filename, index_of_class, m):
	f = open(filename, 'rb')
	dataset = f.readlines()
	length = len(dataset)
	sample=[]
	for i in range(0, length):
		test = dataset[i].split(',')
		for j in range(0, len(test)-1):
			test[j] = float(test[j])
		sample.append(test[:-1])

	s1 = sample[:50]
	s2 = sample[50:100]
	s3 = sample[100:]

	sample1=np.asarray([0])
	for i in range(len(s1)):
		for j in range(0, len(s1[i])):
			m[0][j]+=float(s1[i][j])
		s1 = np.array(s1)
		sample1 = np.concatenate([sample1,s1[i]])
	sample1=np.delete(sample1,0)

	sample2=np.asarray([0])
	for i in range(len(s2)):
		for j in range(0, len(s2[i])):
			m[1][j]+=float(s2[i][j])
		s2 = np.array(s2)
		sample2 = np.concatenate([sample2,s2[i]])
	sample2=np.delete(sample2,0)


	sample3=np.asarray([0])
	for i in range(len(s3)):
		for j in range(0, len(s3[i])):
			m[2][j]+=float(s3[i][j])
		s3 = np.array(s3)
		sample3 = np.concatenate([sample3,s3[i]])
	sample3=np.delete(sample3,0)
	m=m/float(50)
	return sample1, sample2, sample3, m



def calculating_scater_w(sample1, sample2, sample3):

	sample1 = np.transpose(sample1)
	scatter1 = np.dot(sample1, sample1.T)

	sample2 = np.transpose(sample2)
	scatter2 = np.dot(sample2, sample2.T)

	sample3 = np.transpose(sample3)
	scatter3 = np.dot(sample3, sample3.T)
	scatter_w = scatter1+scatter2+scatter3

	return scatter_w


def calculating_scater_b(sample, mean1, mean2, mean3, mean4):

	for i in range(0, 3):
		m[i][0]=m[i][0]-mean1
		m[i][1]=m[i][1]-mean2
		m[i][2]=m[i][2]-mean3
		m[i][3]=m[i][3]-mean4
	

	scatter_b=np.dot(m.T,m)
	scatter_b = scatter_b*50
	return scatter_b

def calculating_lda(scatterw, scatterb, sample):

	eigenvalue, eigenvector = np.linalg.eig(np.linalg.inv(scatterw).dot(scatterb))

	eigenpairs=[]
	for i in range(len(eigenvalue)):
		eigenpairs.append((np.abs(eigenvalue[i]), eigenvector[:,i]))

	sorted(eigenpairs,reverse=True)
	lda = np.hstack((eigenpairs[0][1].reshape(4,1)))
	temp = sample.T
	lda=np.dot(temp,lda)
	
	return lda



def save_plot(x1, x2, x3, name):

	plt.xlabel("LDA")
	class1=plt.plot(x1,	50*[0],'ro',label="Iris-setosa")
	class3=plt.plot(x3,	50*[0],'go',label="Iris-virginica")
	class2=plt.plot(x2,	50*[0],'bo',label="Iris-versicolor")
	
	
	location='lower right'
	plt.legend(loc=location)
	plt.savefig(name)
	plt.close()


def plot(lda):
	x1, x2, x3 = lda[0:50], lda[50:100], lda[100:150]

	save_plot(x1, x2, x3, 'lda.png')


if __name__ == '__main__':

	filename = "iris.data"
	index_of_class=4
	m=np.zeros(12)
	m=m.reshape(3,4)
	sample1, sample2, sample3, m = read_file_load_dataset(filename, index_of_class, m)

	sample = np.concatenate([sample1, sample2, sample3])
	sample = sample.reshape(150, 4)
	sample = np.transpose(sample)

	sample1 = sample1.reshape(50, 4)
	sample2 = sample2.reshape(50, 4)
	sample3 = sample3.reshape(50, 4)


	for i in range(0, 50):
		sample1[i]=sample1[i]-m[0]
		sample2[i]=sample2[i]-m[1]
		sample3[i]=sample3[i]-m[2]

	scatter_w = calculating_scater_w(sample1, sample2, sample3)
	mean1, mean2, mean3, mean4 = np.mean(sample[0]), np.mean(sample[1]), np.mean(sample[2]), np.mean(sample[3]) 
	scatter_b = calculating_scater_b(sample, mean1, mean2, mean3, mean4)

	lda= calculating_lda(scatter_w, scatter_b, sample)

	plot(lda)


