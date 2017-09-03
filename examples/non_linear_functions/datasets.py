import numpy as np

def boolean_function_dataset(function):
	X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	Y = np.array([function(x) for x in X])
	return X, Y

def real_function_dataset(length, function, nb_inputs=2, x_min=0, x_max=1):
	X = np.random.rand(length, nb_inputs) * (x_max - x_min) + x_min
	Y = np.array([function(x) for x in X])
	return X, Y

def or_dataset():
	return boolean_function_dataset(lambda x: x[0] or x[1])

def and_dataset():
	return boolean_function_dataset(lambda x: x[0] and x[1])

def xor_dataset():
	return boolean_function_dataset(lambda x: (x[0] and not x[1]) or (not x[0] and x[1]))

def plane_dataset(length, normal_vector=[1, -1]):
	return real_function_dataset(length, lambda x: (np.dot(x, normal_vector) >= 0))

def disk_dataset(length):
	return real_function_dataset(length, lambda x: ((x[0] - 0.5)**2 + (x[1] - 0.5)**2) <= 0.25, x_min = -0.5, x_max=1.5)