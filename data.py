# -*-coding:UTF-8 -*
import csv

import numpy as np

def import_test_data(filename,nb_samples):
"""
	import {filename} with {nb_samples} as test data
"""
	x = []
	with open(filename, 'rb') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader:
			row = [int(i) for i in row]
			x.append(row)
	return np.array(x[0:nb_samples])


def import_data(directory,is_group,datasize):
"""
	imports {is_group}_{datasize}.csv in {directory} with outputs [1,0] if contains group ({is_group}) and [0,1] {otherwise} 
"""
	inputs = []
	outputs = []
	filename = directory+str(is_group)+'_'+str(datasize)+'.csv'
	with open(filename, 'rb') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader:
			row = [int(i) for i in row]
			inputs.append(row)
			if (is_group):
				outputs.append([1,0])
			else:
				outputs.append([0,1])
	return inputs[0:datasize], outputs[0:datasize]
	
def shuffle_data(inputs,outputs):
"""
	shuffle dataset
"""
	combined = list(zip(inputs, outputs))
	np.random.shuffle(combined)
	inputs[:], outputs[:] = zip(*combined)
	return np.array(inputs), np.array(outputs)
