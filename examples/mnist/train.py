import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../../src')))

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from graph import *
from node import *
from optimization_algorithm import *
from mnist import *

# Create the dataset

def shuffle_dataset(X, Y):
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]

def preprocess_dataset(X, mean=None):
    mean_X = np.mean(X, axis=0) if mean is None else mean
    return (X - mean_X) / 255, mean_X

def one_hot_encode(Y):
    return (np.dot(np.ones((Y.shape[0], 10)), np.diag(np.arange(10))) == Y).astype(float)

# Create the network

def init_function(shape):
    return np.random.rand(*shape)*0.2 - 0.1

def create_fully_connected_network(layers):
    nodes = []
    parameter_nodes = []

    # Input
    input_node = InputNode()
    nodes.append(input_node)

    prev_size = 28*28+1
    cur_input_node = input_node
    for i, size in enumerate(layers):
        # Create a layer
        bias_node = AddBiasNode([(cur_input_node, 0)])
        parameter_node = ParameterNode(init_function((prev_size, size)))
        prod_node = MultiplicationNode([(bias_node, 0), (parameter_node, 0)])
        # Activation function for hidden layers
        if i+1 < len(layers):
            activation_node = TanhNode([(prod_node, 0)])
        # Activation function for the output layer
        else:
            activation_node = SoftmaxNode([(prod_node, 0)])
        # Save the new nodes
        parameter_nodes.append(parameter_node)
        nodes += [bias_node, parameter_node, prod_node, activation_node]
        cur_input_node = activation_node
        prev_size = size + 1
    
    # Expected output
    expected_output_node = InputNode()
    # Cost function
    cost_node = CategoricalCrossEntropyNode([(expected_output_node, 0), (cur_input_node, 0)])

    nodes += [expected_output_node, cost_node]
    return Graph(nodes, [input_node], [cur_input_node, nodes[-7]], [expected_output_node], cost_node, parameter_nodes) 

# Training

def train_and_monitor(monitor=True):
    start_time = time.time()
    t = []
    accuracies_training = []
    accuracies_test = []
    i_batch = 0
    # Optimization algorithm
    sgd = GradientDescent(graph.get_parameter_nodes(), learning_rate)
    for i in range(nb_times_dataset):
        for j in range(0, X.shape[0], batch_size):
            # Train
            graph.propagate([X[j:j+batch_size]])
            cost = graph.backpropagate([ohe_Y[j:j+batch_size]])
            sgd.optimize(batch_size)
            # Monitor
            i_batch += 1
            print('pass: {}/{}, batch: {}/{}, cost: {}'.format(i, nb_times_dataset, \
                    j // batch_size, X.shape[0] // batch_size, cost / batch_size))
            if monitor and (i_batch % 256) == 1:
                t.append(i_batch)
                accuracies_training.append(accuracy(graph, X, Y))
                accuracies_test.append(accuracy(graph, X_test, Y_test))

    print('duration:', time.time() - start_time)
    print('accuracies: ', accuracies_test)
    # Plot the learning curves
    if monitor:
        plt.plot(t, accuracies_training, label='Training set')
        plt.plot(t, accuracies_test, label='Test set')
        plt.xlabel('Number of batches')
        plt.ylabel('Accuracy')
        plt.title('Learning curves')
        plt.legend(loc=4)
        plt.show()

# Accuracy

def get_predicted_class(predicted_y):
    return np.argmax(predicted_y, axis=1)

def accuracy(graph, X, Y):
    predicted_y = graph.propagate([X])[0]
    predicted_class = get_predicted_class(predicted_y)
    return np.sum(Y.flatten() == predicted_class) / Y.shape[0]

# Visualization

def visualize(graph, X, Y, mean, nb_samples=25):
    plt.figure()
    images = (X[:nb_samples]*255 + mean).reshape(nb_samples, 28, 28)
    labels = Y[:nb_samples]
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(5, 5, i+1)
        plt.imshow(image, cmap='Greys', vmin=0, vmax=255, interpolation='none')
        plt.title(str(get_predicted_class(graph.propagate([X[i:i+1]])[0])[0])+ ' ' + str(labels[i]))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()

def display_weights(parameter_nodes, layers):
    plt.figure()
    image_size = 28
    nb_images = layers[0]
    nb_columns = int(np.sqrt(nb_images))
    nb_rows = (nb_images + nb_columns - 1) // nb_columns
    for i, column in enumerate(parameter_nodes[0].w.T):
        # Reshape image
        image = column[1:].reshape((image_size, image_size))
        # Plot
        plt.subplot(nb_rows, nb_columns, i+1)
        plt.imshow(image, cmap='Greys', interpolation='none')
        #plt.title(str(i))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

def display_deep_features(graph, X, Y):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    # Compute the deep features
    features = graph.propagate([X])[1]
    # PCA with 3 components
    features -= np.mean(features, axis=0)
    U, S, V = np.linalg.svd(features)
    points = np.dot(features, V.T[:,:3])
    # Plot points for each digit
    for i in range(10):
        x, y, z = points[Y == i,:].T
        ax1.plot(x, y, 'o', label=str(i))
        ax2.scatter(x, y, z, label=str(i))
    plt.title('Deep features')

if __name__ == '__main__':
    # Prepare dataset
    X, (nb_rows, nb_columns), Y = get_training_set('examples/mnist/mnist')
    X, mean = preprocess_dataset(X)
    X, Y = shuffle_dataset(X, Y)
    Y = Y.reshape((len(Y), 1))
    ohe_Y = one_hot_encode(Y)

    X_test, (_, _), Y_test = get_test_set('examples/mnist/mnist')
    X_test, _ = preprocess_dataset(X_test, mean)
    X_test, Y_test = shuffle_dataset(X_test, Y_test)
    Y_test = Y_test.reshape((len(Y_test), 1))

    # Create the graph
    layers = [10]
    batch_size = 128
    learning_rate = 1
    nb_times_dataset = 1
    graph = create_fully_connected_network(layers)

    # Train
    train_and_monitor(False)

    # Print final accuracy
    print('final accuracy:', accuracy(graph, X_test, Y_test))

    # Interprete
    visualize(graph, X, Y, mean)
    plt.show()
    display_weights(graph.get_parameter_nodes(), layers)
    plt.show()
    display_deep_features(graph, X[:1000], Y[:1000].flatten())
    plt.show()