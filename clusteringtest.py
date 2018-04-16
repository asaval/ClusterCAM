from pyclustering.cluster.optics import optics
from pyclustering.utils import read_sample, draw_clusters
from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES
import matplotlib.pyplot as plt
import numpy as np
import math
import os, sys


def epsilon(x_size, y_size, example, sample, minpts):
"""
	heuristics to get epsilon according to width {x_size}, height {y_size}, grid {example}, list {sample}, MinPts
"""
	grid = np.reshape(example,(x_size,y_size))
	
	# splitting grid
	nx = 10
	ny = 10
	x_step = x_size/nx
	y_step = y_size/ny
	
	# densities for each cell
	cellpop = np.zeros(dtype = np.float32, shape = (nx,ny))
	for y in range(ny):
		for x in range(nx):
			area = grid[x*x_step:(x+1)*x_step,y*y_step:(y+1)*y_step]
			cellpop[x][y] = sum(area.flatten())
	cellpop = cellpop/(x_step*y_step)
	
	
	meanpop = np.mean(cellpop)
	e = 0
	for x in range(nx):
		for y in range(ny):
			if (cellpop[x][y]-meanpop)>0:
				e+=cellpop[x][y]-meanpop
	e = e/(nx*ny)
	density = meanpop/e

	return 3*math.sqrt(minpts/density)
	



# get clusters & noise
def template_clustering(example, sample, w, h):
"""
	outputs clusters & noise. grid {example}, list {sample}
"""
	# max epsilon
	x_size = w
	y_size = h
	maxeps = max(x_size,y_size)
	
	minpts = 12

	optics_instance = optics(sample, maxeps, minpts)
	optics_instance.process()
    
    # extract clusters from ordering
	epsi = epsilon(x_size, y_size, example, sample, minpts)
	optics_instance.extract_clusters(epsi)
	clusters = optics_instance.get_clusters()
	noise = optics_instance.get_noise()
	
	clusters = [[sample[pindex] for pindex in cluster] for cluster in clusters]
	noise = [sample[pindex] for pindex in noise]
	return clusters, noise

