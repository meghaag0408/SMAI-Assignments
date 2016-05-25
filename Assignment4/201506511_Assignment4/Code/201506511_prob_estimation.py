import random
from matplotlib.pyplot import *
import csv
from math import *
import operator
from tabulate import tabulate

# Function to load the dataset and create training sample and test sample
# Divide the dataset into ratio 1:1

def read_file_load_dataset_categorical(filename):
	f = open(filename, 'rb')
	dataset = f.readlines()
	dataset = dataset[1:]
	length = len(dataset)
	random.shuffle(dataset) 
	test, train, training_sample, test_sample =[], [], [], []	
	list_not_to_take = [0, 10, 11, 12, 13, 15, 16, 17, 18, 19]

	#Spliting the dataset into training sample : list of lists
	for i in range(0, length/2):
		train = dataset[i].split(';')
		temp = list(train[len(train)-1])
		temp = temp[:-2]
		temp = "".join(temp)
		train[len(train)-1]=temp
		train_list=[]
		for j in range(len(train)):
			if j not in list_not_to_take:
				train_list.append(train[j])				
		training_sample.append(train_list)
		train=[]
		train_list=[]

	#Spliting the dataset into test sample - with classname(for simplicity): list of lists
	for i in range((length/2), length):
		test = dataset[i].split(';')
		temp = list(test[len(test)-1])
		temp = temp[:-2]
		temp = "".join(temp)
		test[len(test)-1]=temp
		test_list=[]
		for j in range(len(test)):
			if j not in list_not_to_take:
				test_list.append(test[j])		
		test_sample.append(test_list)
		test=[]
		test_list=[]

	return training_sample, test_sample

def read_file_load_dataset_continous(filename):
	f = open(filename, 'rb')
	dataset = f.readlines()
	length = len(dataset)
	random.shuffle(dataset) 
	test, train, training_sample, test_sample =[], [], [], []	
	
	#Spliting the dataset into training sample : list of lists
	for i in range(0, length/2):
		train = dataset[i].split(',')
		train = train[1:]
		for i in range(0, len(train)):
			if train[i]=='?':
				train[i]=5
			train[i]= float(train[i])			
		training_sample.append(train)
		train=[]

	#Spliting the dataset into training sample : list of lists
	for i in range((length/2), length):
		test = dataset[i].split(',')
		test = test[1:]
		for i in range(0, len(test)):
			if test[i]=='?':
				test[i]=5
			test[i]= float(test[i])			
		test_sample.append(test)
		test=[]

	return training_sample, test_sample


def forming_dictionary(training_sample):
	dict_training_sample=[]

	dict={}
	for i in range(len(training_sample[0])-1):
		dict_training_sample.append(dict)


	for j in range(len(dict_training_sample)):
		d_temp={}
		d_temp[1], d_temp[0]  = {}, {}
		d_temp_yes,d_temp_no ={}, {}
	
		for i in range(len(training_sample)):		
			if training_sample[i][j] in d_temp_yes and training_sample[i][-1]=='"yes"':
				d_temp_yes[training_sample[i][j]]+=1
			elif training_sample[i][j] in d_temp_no and training_sample[i][10]=='"no"':
				d_temp_no[training_sample[i][j]]+=1
			else:
				d_temp_yes[training_sample[i][j]]=1
				d_temp_no[training_sample[i][j]]=1

		d_temp[1] = d_temp_yes
		d_temp[0] = d_temp_no
		dict_training_sample[j]=d_temp


	return dict_training_sample


def forming_dictionary_mean_std(training_sample):
	dict_training_sample = {}
	training_sample_2, training_sample_4 =[], []
	
	for i in range(len(training_sample)):
		if training_sample[i][len(training_sample[i])-1]==2.0:
			training_sample_2.append(training_sample[i])
		else:
			training_sample_4.append(training_sample[i])
	
	training_sample_4, training_sample_2 = np.array(training_sample_4), np.array(training_sample_2)
	dict_training_sample={}
	
	l1, l2 =[], []
	for i in range(len(training_sample[0])-1):
		x = training_sample_2[:, i]
		y = training_sample_4[:, i]
		mean1, mean2 = np.mean(x), np.mean(y)
		std1, std2 = np.std(x), np.std(y)
		tup1, tup2 = (mean1, std1),(mean2, std2)
		l1.append(tup1)
		l2.append(tup2)	

	dict_training_sample[2], dict_training_sample[4]=l1, l2
	return dict_training_sample


def calculating_class_label(dict_training_sample, test_sample, no_count, yes_count):	
	y=1
	n=1
	temp1=1
	temp2=1
	prob_yes = float(yes_count)/float(yes_count+no_count)
	prob_no = float(no_count)/float(no_count+yes_count)

	for i in range(4, len(dict_training_sample)):
		if not dict_training_sample[i][1][test_sample[i]]:
			temp1=0
		else:
			temp1 = dict_training_sample[i][1][test_sample[i]]
			temp1 = temp1 / float(yes_count)
		if not dict_training_sample[i][0][test_sample[i]]:
			temp2=0
		else:
			temp2 = dict_training_sample[i][0][test_sample[i]]
			temp2 = temp2/ float(no_count)
		y = y*temp1
		n = n*temp2

	final_prob_yes= float(prob_yes)*float(y)
	final_prob_no = float(prob_no)*float(n)
	prob_no_final_list.append(final_prob_no)
	prob_yes_final_list.append(final_prob_yes)
	if final_prob_yes>final_prob_no:
		return '"yes"'
	else:
		return '"no"'
	
def calculating_class_label_continous(dict_training_sample, test_sample):
	list2 = dict_training_sample[2]
	list4 = dict_training_sample[4]
	p2 = 1.0
	p4 = 1.0
	for i in range(len(test_sample)):
		p2 = p2 * gaussian_function(list2[i][0], list2[i][1] ,test_sample[i])
		p4 = p4 * gaussian_function(list4[i][0], list4[i][1] ,test_sample[i])

	prob_no_final_list.append(p2)
	prob_yes_final_list.append(p4)
	if p2>p4:
		return 2.0
	else:
		return 4.0



def calculate_accuracy(predictions, actual_classes):
	correct=0
	for i in range(len(actual_classes)):
		if actual_classes[i] == predictions[i]:
			correct = correct+1
	accuracy_percentage = (correct/(float(len(actual_classes)))) * 100
	return accuracy_percentage

def calculating_class_probability(training_sample):
	D={}
	for i in range(len(training_sample)):
		if training_sample[i][len(training_sample[i])-1] in D:
			D[training_sample[i][len(training_sample[i])-1]]+=1
		else:
			D[training_sample[i][len(training_sample[i])-1]]=1
	yes_count = D['"yes"']
	no_count =  D['"no"']
	return yes_count, no_count

def confusion_matrix(predictions, actual_classes):
	#Fetching the name of the classes to dictionary and then to the list
	classes={}
	for i in range(len(actual_classes)):
		if actual_classes[i] in classes:			
			classes[actual_classes[i]]= 1
		else:
			classes[actual_classes[i]]= 1
	c =[]
	for i in classes.keys():
		c.append(i)
	length = len(c)	
	

	#Creating confusion matrix as list -> empty list and hence comparing and increasing the count
	confusion_matrix=[]
	for i in range(length):
		for j in range(length):
			confusion_matrix.append(0)

	count = 0
	for i in range(len(actual_classes)):
		for j in range(length):
			for k in range(length):
				if actual_classes[i] == c[j] and predictions[i] == c[k]:
					count = count +1
					confusion_matrix[j*length+k] = confusion_matrix[j*length+k]+1

	#Printing confusion matrix
	if filename == 'wisconsin.data':
		for i in range(length):
			if c[i] == '2':
				c[i] = 'Benign'
			if c[i]=='4':
				c[i] ='Malignant'
	print "\t\t"+'PREDICTED'
	table = []
	
	#Append Classes name
	L=[]
	L.append('\t')
	L.append('\t')
	for i in range(length):
		L.append(c[i])
	table.append(L)

	#Create Empty Table
	L=[]
	for i in range(length):
		for j in range(length+2):
			if i==length/2:
				if j==0:
					L.append('ACTUAL')
				elif j==1:
					L.append(c[i])
				else:
					L.append('\t')
			else:
				if j==1:
					L.append(c[i])
				else:
					L.append('\t')
		table.append(L)
		L=[]

	#Populate value to the confusion matrix/empty table
	value_index=0
	for i in range(1, length+1):
		for j in range(2, length+2):
			table[i][j] = confusion_matrix[value_index]
			value_index+=1

	print tabulate(table, tablefmt="grid")

def gaussian_function(mean,stddev,x):
	temp=float((x-mean))/stddev
	temp=temp*temp*0.5
	b=np.exp(-temp)
	a=float(1)/(stddev*sqrt(2*(float(22)/7)))
	return a*b

def maximum_accuracy_find(accuracy_percentage_list):
	maximum = accuracy_percentage_list[0]
	index = 0
	for i in range(1, len(accuracy_percentage_list)):
		if accuracy_percentage_list[i]>maximum:
			maximum = accuracy_percentage_list[i]
			index = i

	return maximum, index

if __name__ == '__main__':
	print "Enter 1...........categorical data"
	print "Enter 2...........continous data"
	choice = input("Enter Choice\n")
	if choice == 1:
		filename='bank-full.data'
	elif choice==2:
		filename ='wisconsin.data'

	accuracy_percentage_list=[]
	
	for x in range(10):
		prob_yes_final_list=[]
		prob_no_final_list=[]
		print "\n"
		print 'ITERATION NO : ' + repr(x+1)
		if choice==1:
			training_sample, test_sample= read_file_load_dataset_categorical(filename)
			dict_training_sample = forming_dictionary(training_sample)
			prob_yes, prob_no = calculating_class_probability(training_sample)
			print "Probability P(yes) = ", float(prob_yes)/float(prob_no+prob_yes)
			print "Probability P(no) = ", float(prob_no)/float(prob_no+prob_yes)

		elif choice==2:
			training_sample, test_sample= read_file_load_dataset_continous(filename)
			dict_training_sample = forming_dictionary_mean_std(training_sample)

		#print dict_training_sample		
		actual_classes, prediction_list=[],[]
		for i in range(len(test_sample)):
			actual_classes.append(test_sample[i][-1])

		for i in range(len(test_sample)):
			test_sample[i] = test_sample[i][:-1]
			if choice==1:
				prediction = calculating_class_label(dict_training_sample, test_sample[i], prob_no, prob_yes)				
			if choice==2:
				prediction = calculating_class_label_continous(dict_training_sample, test_sample[i])
			prediction_list.append(prediction)
				
		accuracy_percentage = calculate_accuracy(prediction_list, actual_classes)
		accuracy_percentage_list.append(accuracy_percentage)
		prob_no_final_list = np.array(prob_no_final_list)
		prob_yes_final_list = np.array(prob_yes_final_list)
			
		if choice==1:
			print "Average P(xi/yes) * P(yes)= ", np.average(prob_yes_final_list)
			print "Average P(xi/no) * P(no)= ", np.average(prob_no_final_list)
		if choice==2:
			print "Average P(xi/2) * P(2)= ", np.average(prob_no_final_list)
			print "Average P(xi/4) * P(4)= ", np.average(prob_yes_final_list)
		
		print 'Accuracy: ' + repr(accuracy_percentage)
		print
		confusion_matrix(prediction_list, actual_classes)
		print "********************************************"
		print


	print 
	print '=================================================='
	maximum_accuracy, index = maximum_accuracy_find(accuracy_percentage_list)
	print "Maximum Accuracy ", maximum_accuracy
	print "Iteration No of Maximum Accuracy ", index+1
	accuracy_percentage_list = np.array(accuracy_percentage_list)
	print "Mean", np.mean(accuracy_percentage_list)
	print "Standard Deviation", np.std(accuracy_percentage_list)
	print '=================================================='


		
