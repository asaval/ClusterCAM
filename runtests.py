# -*-coding:UTF-8 -*
import csv

import numpy as np
import operator as op

from scipy.special import comb

from cam import *
from clusteringtest import *
from pyclustering.utils import read_sample, draw_clusters

from timeit import default_timer as timer

def import_samples(directory, nb_samples):
"""
	imports all {nb_samples} 'test_{n}_pts.txt' files from {directory}
	'test_{n}_pts.txt' = list of points (coordinates) from sample {n}
	
"""
	samples = []
	for s in range(nb_samples):
		sample = read_sample(directory+'test_'+str(s)+'_pts.txt')
		samples.append(sample)
	return samples
	
	
	

def import_clusters(infos, sample):
"""
	import real clusters & noise from sample using infos
	infos : nb clusters, [nb points for each cluster]
"""  
	nbClusters = infos[0]
	nbPts = infos[1:len(infos)]
	clusters = []
	noise = sample
	for c in range(nbClusters):
		cluster = noise[0:nbPts[c]]
		noise = noise[nbPts[c]:len(noise)]
		clusters.append(cluster)
	return clusters, noise



def ncr(n,r):
    return comb(n,r)
    

def adjusted_rand_index(c1,noise1,c2,noise2):
"""
	Adjusted Rand Index
"""
	if (len(c2)==0):
		if (len(c1)==0):
			return 1
		else:
			return 0
	elif (len(c1)==0):
		return 0
	else:
		c1.append(noise1)
		c2.append(noise2)
		l1 = len(c1)
		l2 = len(c2)
	
		nj = np.zeros(dtype = np.int, shape = (l2))
		snic2 = 0
		snijc2 = 0
		for i in range(l1):
			ni = 0
			si = set(map(tuple,c1[i]))
			for j in range(l2):
				sj = set(map(tuple,c2[j]))
				common = si.intersection(sj)
				nij = len(common)
			
				snijc2 += ncr(nij,2)
			
				ni+=nij
				nj[j]+=nij
			
			snic2 += ncr(ni,2)
		
		snjc2 = 0
		n = 0
		for j in range(l2):
			snjc2 += ncr(nj[j],2)
			n += nj[j]
		return ( snijc2 - (snic2*snjc2)*1.0/ncr(n,2) ) / ( (snic2+snjc2)*1.0/2.0 - (snic2*snjc2)*1.0/ncr(n,2) )




def save_results(directory, algotype, infos):
"""
	saves results {infos} for {algotype} in {directory}
"""
	filename = directory+algotype+'.csv'
	with open(filename, 'a') as f:
		writer = csv.writer(f)
		writer.writerow(infos)





directory = 'tests/'
model_name = 'cam_1000'
nb_samples = 1000


# importing data
x_test = import_test_data(directory+'test.csv',nb_samples) # grids
x_infos = import_test_data(directory+'infos.csv',nb_samples) # infos
samples = import_samples(directory, nb_samples) # lists

# results path
newpath = directory+model_name+"_results_"+str(datetime.now())
if not os.path.exists(newpath): os.makedirs(newpath)

w = 100
h = 100

cam_model = load_model(model_name+'.h5')

for s in range(nb_samples):
	sample = samples[s]
	
	# real clustering
	real_clusters, real_noise = import_clusters(x_infos[s],sample)
	
	# test OPTICS
	start = timer()
	optics_clusters, optics_noise = template_clustering(x_test[s], sample, w, h)
	end = timer()
	ARI_opt = adjusted_rand_index(real_clusters,real_noise,optics_clusters,optics_noise)
	save_results(newpath+'/','OPTICS',[len(optics_clusters),ARI_opt,end-start])

	#test CLUSTERCAM
	start = timer()
	cam_clusters, cam_noise = process_example(cam_model, sample, x_test[s], w, h)
	end = timer()
	ARI_cam = adjusted_rand_index(real_clusters,real_noise,cam_clusters,cam_noise)
	save_results(newpath+'/','CAM',[len(cam_clusters),ARI_cam,end-start,len(cam_noise)])
