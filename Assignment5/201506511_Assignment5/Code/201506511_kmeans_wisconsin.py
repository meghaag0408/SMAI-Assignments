import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from tabulate import tabulate
import math


def read_file_load_dataset(filename, index_of_class):
	with open(filename,'rb') as f:
		reader=csv.reader(f)
		sample=list(reader)
	count_class_2=0
	count_class_4=0
	random.shuffle(sample)
	for i in range(0, len(sample)):
		temp=[]
		for j in range(1, len(sample[i])):
			if sample[i][j]=='?':
				sample[i][j]=5
			temp.append(int(sample[i][j]))
		if sample[i][-1]=='2':
			count_class_2+=1
		if  sample[i][-1]=='4':
			count_class_4+=1
		sample[i]=temp
	for i in range(0, len(sample)):
		sample[i].extend('0')
	return sample, count_class_2, count_class_4

def calculating_distances(data, m1, m2):
	dist1, dist2=-1.0, -1.0
	mindist=10000.0
	temp1, temp2 =m1[:-2], m2[:-2]
	temp1 = np.asarray(temp1)
	temp2 = np.asarray(temp2)

	dist1 = data-temp1
	dist2 = data-temp2
	
	dist1=np.dot(dist1, dist1)
	dist2=np.dot(dist2, dist2)
	return dist1, dist2

def finding_min_distance(dist1, dist2,  sample):
	mindist=dist1
	sample[-1]='1'
	if mindist>dist2:
		sample[-1]='2'
		mindist=dist2

	return mindist, sample


def assign_clusters(l, m1, m2, previousm1, previousm2, count_class_2, count_class_4):
	no_of_iterations=0
	flag=0

	while flag!=1:
		no_of_iterations+=1
		for k in range(len(l)):						
			data=l[k]
			data=data[:-2]
			data = np.asarray(data)
			dist1, dist2= calculating_distances(data, m1, m2)
			mindist, l[k] = finding_min_distance(dist1, dist2, l[k])	
		print dist1, dist2
		cnt1, cnt2 = 0,0
		m=np.zeros(18).reshape(2,9)		
		for data in l:		
			if data[-1]=='1':
				for j in range(len(data)-2):
					m[0][j]=m[0][j]+data[j]
				cnt1+=1
			elif data[-1]=='2':
				for j in range(len(data)-2):
					m[1][j]=m[1][j]+data[j]
				cnt2+=1
		m[0], m[1] =(m[0]/cnt1), (m[1]/cnt2)
		for i in range(0, 9):
			m1[i], m2[i] =m[0][i], m[1][i] 	
		if previousm1==m1 and previousm2==m2:
			flag=1
		
		previousm1, previousm2  =m1[:], m2[:]	
	return no_of_iterations, l

def confusion_matrix_calculation(actual_classes, predictions):
	#Fetching the name of the classes to dictionary and then to the list
	c = ['2', '4']	
	confusion_matrix={'1':[0,0,0], '2':[0,0,0]}
	class_value=[0, 0, 0]
	for j in range(len(predictions)):
		for i in confusion_matrix.keys():
			if i==predictions[j]:
				if actual_classes[j]==c[0]:
					confusion_matrix[i][0]+=1
					confusion_matrix[i][2]+=1
				elif actual_classes[j]==c[1]:
					confusion_matrix[i][1]+=1
					confusion_matrix[i][2]+=1
	
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
				tij = confusion_matrix['1'][j] + confusion_matrix['2'][j]
		purity=purity+(float(max)/float(confusion_matrix[i][2]))
		den = float(confusion_matrix[i][2])+float(tij)
		num = 2 * float(max)
		f_measure+=float(num)/float(den)

	purity = purity/2.0	
	f_measure = f_measure/2.0
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
		temp = temp/math.factorial(max_value-2)
		temp=temp/2.0
		N_in=N_in+temp	
	N_out=(n[0]*n[1])			
	num=float(w_in)/float(N_in)
	den=float(w_out)/float(N_out)
	beta_cv=float(num)/float(den)

	#Finding normalised cut
	normalised_cut=1/float(float(w_in)/float(w_out)+1)
	
	return beta_cv, normalised_cut

if __name__ == '__main__':

	filename = "breast-cancer-wisconsin.data"
	index_of_class=4
	sample, count_class_2, count_class_4 = read_file_load_dataset(filename, index_of_class)

	#Forming Clusters - Randomly initialise K (3 here) clusters
	m1, m2 = sample[0], sample[1]
	m1[-1], m2[-1]= '1', '2'
	previous_m1, previous_m2 = m1[:], m2[:]
	random.shuffle(sample)
	sample1 = np.array(sample)
	#Assigning Clusters
	no_of_iterations, sample = assign_clusters(sample, m1, m2, previous_m1, previous_m2, count_class_2, count_class_4)
	
	print
	print "No of Iterations" , no_of_iterations
	print

	#Confusion Matrix
	sample2 = np.array(sample)
	actual_classes = sample1[:,-2]
	predictions = sample2[:, -1]
	
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
	




