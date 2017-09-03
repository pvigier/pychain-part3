import os
import numpy as np
import matplotlib.pyplot as plt

def bytes_to_int(values):
	result = 0
	for value in values:
		result = result * 256 + value
	return result

def get_images(path):
	f = open(path, 'rb')
	values = f.read()
	nb_images = bytes_to_int(values[4:8])
	nb_rows = bytes_to_int(values[8:12])
	nb_columns = bytes_to_int(values[12:16])
	pixels = np.fromstring(values[16:], dtype=np.uint8)
	images = pixels.reshape((nb_images, nb_rows * nb_columns))
	return images, (nb_rows, nb_columns)

def get_labels(path):
	f = open(path, 'rb')
	values = f.read()
	nb_items = bytes_to_int(values[4:8])
	labels = np.fromstring(values[8:], dtype=np.uint8)
	return labels
	
def get_training_set(path_to_folder='.'):
	return (*get_images(os.path.join(path_to_folder, 'train-images.idx3-ubyte')), 
		    get_labels(os.path.join(path_to_folder, 'train-labels.idx1-ubyte')))

def get_test_set(path_to_folder='.'):
	return (*get_images(os.path.join(path_to_folder, 't10k-images.idx3-ubyte')), 
		    get_labels(os.path.join(path_to_folder, 't10k-labels.idx1-ubyte')))

def visualize(images, image_shape, labels, nb_samples=25):
	indices = np.random.randint(0, images.shape[0], nb_samples)
	images = images[indices].reshape((nb_samples, image_shape[0], image_shape[1]))
	labels = labels[indices]
	for i, (image, label) in enumerate(zip(images, labels)):
		plt.subplot(5, 5, i+1)
		plt.imshow(image, cmap='Greys', vmin=0, vmax=255, interpolation='none')
		plt.title(label)
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	visualize(*get_training_set('examples/mnist/mnist'))
	visualize(*get_test_set('examples/mnist/mnist'))