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

def calculating_class_label_ques1(dict_training_sample, test_sample, no_count, yes_count, x):	
	y=1
	n=1
	temp1=1
	temp2=1
	prob_yes = float(yes_count)/float(yes_count+no_count)
	prob_no = float(no_count)/float(no_count+yes_count)
	print
	print "*************TEST SAMPLE NO",
	print x, "***********************"
	print test_sample
	test_sample = test_sample[:-1]


	for i in range(len(dict_training_sample)):
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

		print "P(x/yes) = ", temp1, 
		print "|&|",
		print "P(x/no) = ", temp2
		y = y*temp1
		n = n*temp2
		print "YES = ", y, 
		print "",
		print "NO = ", n
		print

	final_prob_yes= float(prob_yes)*float(y)
	final_prob_no = float(prob_no)*float(n)
	prob_no_final_list.append(final_prob_no)
	prob_yes_final_list.append(final_prob_yes)
	if final_prob_yes>final_prob_no:
		return '"yes"'
	else:
		return '"no"'

def calculating_class_label_continous_ques1(dict_training_sample, test_sample, x):
	list2 = dict_training_sample[2]
	list4 = dict_training_sample[4]
	p2 = 1.0
	p4 = 1.0
	print
	print "*************TEST SAMPLE NO",
	print x, "***********************"
	print test_sample
	test_sample = test_sample[:-1]
	for i in range(len(test_sample)):
		temp1 = gaussian_function(list2[i][0], list2[i][1] ,test_sample[i])
		temp2 = gaussian_function(list4[i][0], list4[i][1] ,test_sample[i])
		print i+1, "Mean(2) = ", list2[i][0], "  Stddev(2) = ", list2[i][1] 
		print "Mean(4) = ", list4[i][0], "  Stddev(4) = ", list4[i][1] 
		print "P(x/2) = ", temp1, 
		print "|&|",
		print "P(x/4) = ", temp2
		p2 = p2 * temp1
		p4 = p4 * temp2
		print "Cumltive. 2 = ", p2, 
		print "",
		print "Cumltive 4 = ", p4
		print
	prob_no_final_list.append(p2)
	prob_yes_final_list.append(p4)
	if p2>p4:
		return 2.0
	else:
		return 4.0
	
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



def calculate_accuracy(predictions, actual_classes, test_sample):
	correct=0
	test_sample_list=[]
	prediction_wrong_sample_list=[]
	count=0
	for i in range(len(actual_classes)):
		if actual_classes[i] == predictions[i]:
			correct = correct+1
		else:
			if count<3:
				test_sample_list.append(test_sample[i])
				prediction_wrong_sample_list.append(predictions[i])
				count=count+1

	accuracy_percentage = (correct/(float(len(actual_classes)))) * 100
	return accuracy_percentage, test_sample_list, prediction_wrong_sample_list

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



def gaussian_function(mean,stddev,x):
	temp=float((x-mean))/stddev
	temp=temp*temp*0.5
	b=np.exp(-temp)
	a=float(1)/(stddev*sqrt(2*(float(22)/7)))
	return a*b

def maximum_accuracy_find(accuracy_percentage_list):
	maximum = accuracy_percentage_list[0]
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
	
	for x in range(1):
		prob_yes_final_list=[]
		prob_no_final_list=[]
		print
		if choice==1:
			training_sample, test_sample= read_file_load_dataset_categorical(filename)
			dict_training_sample = forming_dictionary(training_sample)
			prob_yes, prob_no = calculating_class_probability(training_sample)
	
		elif choice==2:
			training_sample, test_sample= read_file_load_dataset_continous(filename)
			dict_training_sample = forming_dictionary_mean_std(training_sample)

		#print dict_training_sample		
		actual_classes, prediction_list=[],[]
		for i in range(len(test_sample)):
			actual_classes.append(test_sample[i][-1])

		for i in range(len(test_sample)):
			test_sample_new = test_sample[i][:-1]
			if choice==1:
				prediction = calculating_class_label(dict_training_sample, test_sample_new, prob_no, prob_yes)				
			if choice==2:
				prediction = calculating_class_label_continous(dict_training_sample, test_sample_new)
			prediction_list.append(prediction)
				
		accuracy_percentage, test_sample_list, prediction_wrong_sample_list = calculate_accuracy(prediction_list, actual_classes, test_sample)

	print "***********************************QUESTION1**************************************"

	for i in range(len(test_sample_list)):
		test_sample_new = test_sample_list[i]
		if choice==1:
				prediction = calculating_class_label_ques1(dict_training_sample, test_sample_new, prob_no, prob_yes, i+1)				
		if choice==2:
				prediction = calculating_class_label_continous_ques1(dict_training_sample, test_sample_new, i+1)
		print "PREDICTION = ", prediction
	
