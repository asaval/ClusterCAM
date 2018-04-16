from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
import h5py
from keras.optimizers import SGD


def basic_cnn(width, height):
"""
	outputs basic CNN (2 conv layers, 1 maxpool)
"""
	model = Sequential()
	model.add(Convolution2D(10, (3, 3), activation='relu', input_shape=(width,height,1), name='basic1'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=2))
	model.add(Convolution2D(20, (3, 3), activation='relu', name = 'basic2'))
	return model


# global average pooling
def global_average_pooling(x):
	return K.mean(x, axis = (1, 2)) # average on width & height

def global_average_pooling_shape(input_shape): # input shape : NWHC
	return (input_shape[0],input_shape[3])
	

def get_model(width, height):
"""
	outputs our model according to {width} and {height}
"""
	model = basic_cnn(width, height)
	model.add(Lambda(global_average_pooling,output_shape=global_average_pooling_shape))
	model.add(Dense(2, activation = 'softmax', init='uniform'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
	return model

def get_output_layer(model, layer_name):
"""
	output layer of the model
"""
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer
