import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from tabulate import tabulate
import math


def read_file_load_dataset(filename, index_of_class):
	with open('iris.data','rb') as f:
		reader=csv.reader(f)
		sample=list(reader)
	random.shuffle(sample)
	for i in range(0, len(sample)):
		for j in range(0, len(sample[i])-1):
			sample[i][j] = float(sample[i][j])
	for i in range(0, len(sample)):
		sample[i].extend('0')

	return sample

def calculating_distances(data, m1, m2, m3):
	dist1, dist2, dist3 =-1.0, -1.0, -1.0
	mindist=10000.0
	temp1, temp2, temp3 =m1[:-2], m2[:-2], m3[:-2]
	temp1 = np.asarray(temp1)
	temp2 = np.asarray(temp2)
	temp3 = np.asarray(temp3)

	dist1 = data-temp1
	dist2 = data-temp2
	dist3 = data-temp3

	dist1=np.dot(dist1, dist1)
	dist2=np.dot(dist2, dist2)
	dist3=np.dot(dist3, dist3)
	return dist1, dist2, dist3

def finding_min_distance(dist1, dist2, dist3, sample):
	mindist=dist1
	sample[-1]='1'
	if mindist>dist2:
		sample[-1]='2'
		mindist=dist2
		
	if mindist>dist3:		
		sample[-1]='3'
		mindist=dist3

	return mindist, sample


def assign_clusters(l, m1, m2, m3, previousm1, previousm2, previousm3):
	no_of_iterations=0
	flag=0

	while flag!=1:
		no_of_iterations+=1
		for k in range(len(l)):						
			data=l[k]
			data=data[:-2]
			data = np.asarray(data)
			dist1, dist2, dist3 = calculating_distances(data, m1, m2, m3)
			mindist, l[k] = finding_min_distance(dist1, dist2, dist3, l[k])	
		print dist1, dist2, dist3
		cnt1, cnt2, cnt3 = 0,0,0 
		m=np.zeros(12).reshape(3,4)		
		for data in l:		
			if data[-1]=='1':
				for j in range(len(data)-2):
					m[0][j]=m[0][j]+data[j]
				cnt1+=1
			elif data[-1]=='2':
				for j in range(len(data)-2):
					m[1][j]=m[1][j]+data[j]
				cnt2+=1
			else:
				for j in range(len(data)-2):
					m[2][j]=m[2][j]+data[j]
				cnt3+=1
		m[0], m[1], m[2] =(m[0]/cnt1), (m[1]/cnt2), (m[2]/cnt3)	
		for i in range(0, 4):
			m1[i], m2[i], m3[i] =m[0][i], m[1][i], m[2][i]		
		if previousm1==m1 and previousm2==m2 and previousm3==m3:
			flag=1
		
		previousm1, previousm2, previousm3 =m1[:], m2[:], m3[:]	
	return no_of_iterations, l

def plot(sample):
	# Plot between sepal length and petal length
	x1, x2, x3, y1, y2, y3 = [], [] ,[] , [], [], []

	for i in range(len(sample)):
		if sample[i][-1]=='1':
			x1.append(sample[i][1])
			y1.append(sample[i][3])
		elif sample[i][-1]=='2':
			x2.append(sample[i][1])
			y2.append(sample[i][3])
		else:
			x3.append(sample[i][1])
			y3.append(sample[i][3])
	plt.xlabel("Sepal Width")
	plt.ylabel("Petal Width")
	plt.plot(x1,y1,'ro',label="1")
	plt.plot(x2,y2,'bo',label="2")
	plt.plot(x3,y3,'go',label="3")
	location = "upper left"
	plt.legend(loc=location)
	plt.show()
	plt.savefig("clustering_irs.png")
	plt.close()

def confusion_matrix_calculation(actual_classes, predictions):
	#Fetching the name of the classes to dictionary and then to the list
	c = ['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']	
	confusion_matrix={'1':[0,0,0,0], '2':[0,0,0,0], '3':[0,0,0,0]}
	class_value=[0, 0, 0]
	for j in range(len(predictions)):
		for i in confusion_matrix.keys():
			if i==predictions[j]:
				if actual_classes[j]==c[0]:
					confusion_matrix[i][0]+=1
					confusion_matrix[i][3]+=1
				elif actual_classes[j]==c[1]:
					confusion_matrix[i][1]+=1
					confusion_matrix[i][3]+=1
				elif actual_classes[j]==c[2]:
					confusion_matrix[i][2]+=1
					confusion_matrix[i][3]+=1
	
	return confusion_matrix
	
def external_measures(confusion_matrix):
	purity=0.0
	f_measure=0.0

	#Calculating Purity and F-Measure
	for i in confusion_matrix.keys():
		max=0.0
		for j in range(len(confusion_matrix[i])-1):
			if confusion_matrix[i][j]>max:
				max = confusion_matrix[i][j]
				tij = confusion_matrix['1'][j] + confusion_matrix['2'][j] + confusion_matrix['3'][j]
		purity=purity+(float(max)/float(confusion_matrix[i][3]))
		den = float(confusion_matrix[i][3])+float(tij)
		num = 2 * float(max)
		f_measure+=float(num)/float(den)

	purity = purity/3.0	
	f_measure = f_measure/3.0
	return purity, f_measure

def internal_measures(confusion_matrix, sample):
	beta_cv=0.0
	normalised_cut=0.0
	w_in, w_out= 0.0, 0.0

	for datai in sample:
		for dataj in sample:
			c1, c2 =datai[:-2], dataj[:-2]
			c1, c2 = np.asarray(c1), np.asarray(c2)
			c = c1-c2
			dist=np.dot(c,c)
			dist=math.sqrt(dist)
			if datai[-1]==dataj[-1]:			
				w_in+=dist
			else:
				w_out+=dist	

	w_in=w_in /2.0
	w_out=w_out/2.0

	#Finding number of distinct inter/intra cluster edges
	N_in, N_out=0.0, 0.0
	n=[]
	for i in confusion_matrix.keys():
		max_value=0.0
		for j in range(len(confusion_matrix[i])-1):
			if confusion_matrix[i][j]>max_value:
				max_value = confusion_matrix[i][j]
		n.append(max_value)
		temp = math.factorial(max_value)
		temp = temp/float(math.factorial(max_value-2))
		temp/=2.0
		N_in=N_in+temp	
	sum=0.0	
	for i in range(len(n)-1):
		sum = sum+n[i]*n[i+1]
	N_out=sum+(n[0]*n[2])			
	num=float(w_in)/float(N_in)
	den=float(w_out)/float(N_out)
	beta_cv=float(num)/float(den)

	#Finding normalised cut
	normalised_cut=1/float(float(w_in)/float(w_out)+1)
	
	return beta_cv, normalised_cut

if __name__ == '__main__':

	filename = "iris.data"
	index_of_class=4
	sample = read_file_load_dataset(filename, index_of_class)
	sample1 = np.array(sample)
	labels = sample1[:,-1]

	#Forming Clusters - Randomly initialise K (3 here) clusters
	m1, m2, m3 = sample[0], sample[1], sample[2]
	m1[-1], m2[-1], m3[-1] = '1', '2', '3'
	previous_m1, previous_m2, previous_m3 = m1[:], m2[:], m3[:]
	random.shuffle(sample)

	#Assigning Clusters
	no_of_iterations, sample = assign_clusters(sample, m1, m2, m3, previous_m1, previous_m2, previous_m3)
	print
	print "No of Iterations" , no_of_iterations
	print
	
	#Plotting
	plot(sample)
	
	#Confusion Matrix
	sample1 = np.array(sample)
	actual_classes = sample1[:,-2]
	predictions = sample1[:, -1]

	conf = confusion_matrix_calculation(actual_classes, predictions)

	print "Confusion Matrix"
	print conf
	print

	purity, f_measure = external_measures(conf)
	beta_cv, normalised_cut = internal_measures(conf, sample)

	print '***********External Measures***************'
	print "PURTY............: ", purity
	print "F-Measure........: ", f_measure
	print
	print '***********Internal Measures***************'
	print "Beta-CV..........: ", beta_cv
	print "Normalised Cut...: ", normalised_cut
	print '*******************************************'
	print
	




