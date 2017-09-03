import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../../src')))

import numpy as np
import matplotlib.pyplot as plt
from graph import *
from node import *
from optimization_algorithm import *
from datasets import *

# Create the network

def create_fully_connected_network(layers):
    nodes = []
    parameter_nodes = []

    # Input
    input_node = InputNode()
    nodes.append(input_node)

    cur_input_node = input_node
    prev_size = 3
    for i, size in enumerate(layers):
        # Create a layer
        bias_node = AddBiasNode([(cur_input_node, 0)])
        parameter_node = ParameterNode(np.random.rand(prev_size, size)*1-0.5)
        prod_node = MultiplicationNode([(bias_node, 0), (parameter_node, 0)])
        # Activation function for hidden layers
        if i+1 < len(layers):
            # Tanh
            activation_node = TanhNode([(prod_node, 0)])
            # Relu
            #activation_node = ReluNode([(prod_node, 0)])
        # Activation function for the output layer
        else:
            activation_node = SigmoidNode([(prod_node, 0)])
        # Save the new nodes
        parameter_nodes.append(parameter_node)
        nodes += [bias_node, parameter_node, prod_node, activation_node]
        cur_input_node = activation_node
        prev_size = size + 1

    # Expected output
    expected_output_node = InputNode()
    # Cost function
    cost_node = BinaryCrossEntropyNode([(expected_output_node, 0), (cur_input_node, 0)])

    nodes += [expected_output_node, cost_node]
    return Graph(nodes, [input_node], [cur_input_node, nodes[-7]], [expected_output_node], cost_node, parameter_nodes)

# Interpretation

def visualize(graph, n, x1_min=-0.5, x1_max=1.5, x2_min=-0.5, x2_max=1.5):
    x1s, x2s = np.linspace(x1_min, x1_max, n), np.linspace(x2_min, x2_max, n)
    x1s, x2s = np.meshgrid(x1s, x2s)
    X = np.concatenate((x1s.reshape(x1s.size, 1), x2s.reshape(x2s.size, 1)), axis=1)
    Y = graph.propagate([X])[0].reshape(n, n)
    plt.imshow(Y, extent=[x1_min, x1_max, x2_min, x2_max], vmin=0, vmax=1, origin='lower', cmap='jet')
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Network fully-connected {}'.format(layers))

def display_deep_features(graph, X, Y):
    plt.figure()
    # Compute the deep features
    features = graph.propagate([X])[1]
    # PCA with 2 components
    features -= np.mean(features, axis=0)
    U, S, V = np.linalg.svd(features)
    points = np.dot(features, V.T[:,:2])
    # Plot points for each digit
    for i in range(2):
        x, y = points[Y == i,:].T
        plt.plot(x, y, 'o', label=str(i))
    plt.legend()
    plt.title('Deep features')


if __name__ == '__main__':
    # XOR dataset
    X, Y = xor_dataset()
    # Disk dataset
    #X, Y = disk_dataset(200)
    # Plot the dataset
    plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'bo')
    plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'ro')
    Y = Y.reshape((len(Y), 1))

    # Create the graph and initialize the optimization algorithm
    layers = [4, 4, 1]
    graph = create_fully_connected_network(layers)
    sgd = GradientDescent(graph.get_parameter_nodes(), 0.1)
    # Train
    t = []
    costs = []
    nb_passes = 10000
    for i_pass in range(nb_passes):
        # Propagate, backpropagate and optimize
        graph.propagate([X])
        cost = graph.backpropagate([Y]) / X.shape[0]
        sgd.optimize(X.shape[0])
        # Save the cost
        t.append(i_pass)
        costs.append(cost)
        print(cost)

    # Visualize the frontier
    visualize(graph, 100)
    # Plot the cost
    plt.figure()
    plt.plot(t, costs)
    plt.xlabel('Number of passes')
    plt.ylabel('Cost')
    plt.title('Network fully-connected {}'.format(layers))
    # Display the deep features
    display_deep_features(graph, X, Y.flatten())

    plt.show()
