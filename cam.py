# -*-coding:UTF-8 -*
import csv

from keras.models import *
from keras.callbacks import *
import keras.backend as K
from model import *
from data import *
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import os, sys

from pyclustering.utils import read_sample, draw_clusters

from timeit import default_timer as timer


def train(traindir, output_path, width, height, datasize):
"""
	Builds the network according to the width & height of the input scene
	Trains it using the data in {traindir} directory, with {datasize} examples
	Outputs the trained network in {output_path} h5 file.
"""
	model = get_model(width, height) # builds model
	
	# import & shuffle data from directory {traindir}
	groupin, groupout = import_data(traindir, 1, datasize) # (e.g. from 1_1000.csv)
	noisein, noiseout = import_data(traindir,0, datasize)
	x, y = shuffle_data(groupin+noisein, groupout+noiseout)
	
	# reshaping data (NHWC Nb samples, Height, Width, Channels, because we're using TensorFlow as backend)
	x = x.reshape(x.shape[0], height, width, 1)
	x = x.astype('float32')

	# training
	checkpoint_path="checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
	checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
	model.fit(x, y, nb_epoch=20, batch_size=32, validation_split=0.2, verbose=1, callbacks=[checkpoint])
	
	# saving model
	model.save(output_path)




def connected_sets(cam, csize):
"""
	returns the connected sets of the class activation map {cam} with size {csize}
"""
	threshold = 0.2
	cam[np.where(cam < threshold)] = 0
	cam[np.where(cam >= threshold)] = 1
	
	cam = cv2.resize(cam, (csize,csize))
	
	sets = {} # connected sets
	
	for y in range(csize):
		for x in range(csize):
			if (cam[y][x]>0):
				neighbours = []
				if (y-1>=0):
					neighbours.append((y-1,x))
					if (x-1>=0):
						neighbours.append((y-1,x-1))
					if (x+1<csize):
						neighbours.append((y-1,x+1))
				if (x-1>=0):
					neighbours.append((y,x-1))
								
				nsets = set()
				for numset, value in sets.items():
					for n in neighbours:
						if n in value:
							nsets.add(numset)
						
				if (len(nsets)==0): # new connected set
					if (len(sets)==0):
						numset = 1
					else:
						numset = max(sets)+1
					sets[numset] = [(y,x)]
					
				else:			
					if (len(nsets)==1):
						numset = min(nsets)
					else:
						numset = min(nsets)
						nsets.remove(numset)
						for s in nsets:
							sets[numset] = sets[numset] + sets[s]
							del sets[s]
					sets[numset].append((y,x))
	return sets



def process_example(model, sample, example, width, height):
"""
	processes one example (grid : {example}, list : {sample}) with {model}
"""
	# reshaping {example} -> grid -> network input
	original_img = np.reshape(example,(width,height))
	x = original_img.reshape(1, height, width, 1)
	x = x.astype('float32')
	
	class_weights = model.layers[-1].get_weights()[0] # weights from last layer (GAP)
	final_conv_layer = get_output_layer(model, 'basic2') # CNN output
	
	# outputs from last CNN layer & softmax layer
	get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
	[conv_outputs, predictions] = get_output([x])
	conv_outputs = conv_outputs[0, :, :, :]
	pred = np.argmax(predictions) # prediction (group/no group)
	
	# generating class activation map
	cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
	for i, w in enumerate(class_weights[:,pred]):
		cam += w * conv_outputs[:, :, i]
	cam /= np.max(cam) # heatmap
	
	if (pred==0): # network said "there's a group"
		csize=conv_outputs.shape[0:2][0]
		ratio = (height*1.0)/(csize*1.0) # ratio image size / cam size
		sets = connected_sets(cam,csize)
		clusters, noise = get_clusters(sample, sets, ratio)
	else:
		clusters = []
		noise = sample

	return clusters, noise
	
	
	
def get_clusters(sample, sets, ratio):
"""
	outputs clusters from database {sample} knowing connected sets {sets} with {ratio} realsize/camsize
"""
	clustersdict = {}
	noise = sample
	for ns, s in sets.items():
		delnoise = []
		for p in noise:
			p2 = [x/ratio for x in p]
			if len([pt for pt in sets[ns] if ( (pt[0]==int(p2[1])) and (pt[1]==int(p2[0])) )])!=0:
				delnoise.append(p)
				if ns in clustersdict:
					clustersdict[ns].append(p)
				else:
					clustersdict[ns] = [p]
		for p in delnoise:
			if p in noise:
				noise.remove(p)
	
	clusters = clustersdict.values()
	return clusters, noise


#w = 100
#h = 100
#traindirectory = 'train/'
#nb_samples = 1000
#output_path = 'cam_'+str(nb_samples)+'.h5'
#train(traindirectory, output_path, w, h, nb_samples)
