import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../../src')))

import pickle
import time
import matplotlib.pyplot as plt
import reber
import symmetrical_reber
from graph import *
from node import *
from optimization_algorithm import *

letters = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

# Parameters
learning_rate = 0.01
threshold = 0.3

# Architecture
nb_hidden_neurons = 4

# Reber
automaton = reber.create_automaton()
# Symmetrical Reber
#automaton = symmetrical_reber.create_automaton()

# Create graph

def unfold_graph(n, w1, w2):
    # Initialize containers
    nodes = []
    input_nodes = []
    output_nodes = []
    expected_output_nodes = []
    cost_nodes = []
    parameter_nodes = []
    hidden_nodes = []

    # Initialization of the hidden state
    h_init = InputNode()
    nodes.append(h_init)
    input_nodes.append(h_init)

    # Unfolding
    h = h_init
    for _ in range(n):
        # Input
        x = InputNode()
        total_in = ConcatenationNode([(x, 0), (h, 0)])
        # Hidden layer
        param1 = ParameterNode(w1)
        mult1 = MultiplicationNode([(total_in, 0), (param1, 0)])
        h = TanhNode([(mult1, 0)])
        # Output layer
        param2 = ParameterNode(w2)
        mult2 = MultiplicationNode([(h, 0), (param2, 0)])
        out = SoftmaxNode([(mult2, 0)])
        # Cost
        y = InputNode()
        cost = CategoricalCrossEntropyNode([(y, 0), (out, 0)])

        # Add these nodes to the lists
        nodes += [x, total_in, param1, mult1, h, param2, mult2, out, y, cost]
        input_nodes.append(x)
        output_nodes.append(out)
        expected_output_nodes.append(y)
        cost_nodes.append((cost, 0))
        parameter_nodes += [param1, param2]
        hidden_nodes.append(h)

    total_cost = SumNode(cost_nodes)
    return Graph(nodes, input_nodes, output_nodes+hidden_nodes, expected_output_nodes, total_cost, parameter_nodes)

# Training

def string_to_sequence(string):
    # One-hot encode the string
    sequence = []
    for letter in string:
        x = np.zeros((1, len(letters)))
        x[0, letter_to_index[letter]] = 1
        sequence.append(x)
    return sequence

def train_reber(w1, w2, N):
    start_time = time.time()
    t = []
    accuracies = []
    cost = 0
    for i in range(N):
        # Generate a new sequence
        string = automaton.generate()
        sequence = string_to_sequence(string)
        # Unfold the graph
        graph = unfold_graph(len(string)-1, w1, w2)
        # Propagate and backpropagate
        graph.propagate([np.zeros((1, nb_hidden_neurons))] + sequence[:-1])
        cost += graph.backpropagate(sequence[1:])
        # Optimize the weights
        sgd = GradientDescent(graph.get_parameter_nodes(), learning_rate)
        sgd.optimize()
        # Print metrics
        if (i+1) % 100 == 0:
            t.append(i)
            accuracies.append(accuracy(1000))
            print('strings: {}/{}, cost: {:.2f}, acc: {:.2f}'.format(i+1, N, cost / 100, accuracies[-1]))
            cost = 0

    # Plot the learning curve
    plt.plot(t, accuracies)
    plt.xlabel('Number of strings')
    plt.ylabel('Accuracy')
    plt.title("Learning curve with {} hidden neurons ({:.2f}s)".format(nb_hidden_neurons, time.time() - start_time))
    plt.show()

# Accuracy

def predict_correctly(string):
    sequence = string_to_sequence(string)
    graph = unfold_graph(len(string)-1, w1, w2)
    output = graph.propagate([np.zeros((1, nb_hidden_neurons))] + sequence[:-1])
    cur_state = automaton.start
    for i, (x, y) in enumerate(zip(sequence[:-1], output)):
        cur_state = cur_state.next(string[i])
        predicted_transitions = {letters[j] for j, activated in enumerate(y[0].flatten() > threshold) if activated}
        if set(predicted_transitions) != set(cur_state.transitions.keys()):
            return False
    return True

def accuracy(N):
    return np.mean([predict_correctly(automaton.generate()) for _ in range(N)])

# Interpretation

def display_output(w1, w2):
    string = automaton.generate()
    print(string)
    sequence = string_to_sequence(string)
    graph = unfold_graph(len(string)-1, w1, w2)
    output = graph.propagate([np.zeros((1, nb_hidden_neurons))] + sequence[:-1])
    for letter, y in zip(string[:-1], output):
        print(letter, [(l, p) for l, p in zip(letters, y[0].flatten())])

def plot_deep_states(w1, w2):
    deep_states = []
    states = []
    for i in range(500):
        string = automaton.generate()
        sequence = string_to_sequence(string)
        graph = unfold_graph(len(string)-1, w1, w2)
        output = graph.propagate([np.zeros((1, nb_hidden_neurons))] + sequence[:-1])
        deep_states += output[len(string)-1:]
        cur_state = automaton.start
        for i in range(len(sequence)-1):
            cur_state = cur_state.next(string[i])
            states.append(cur_state)
    # Reshape data
    X = np.array(deep_states)
    X = X.reshape(X.shape[0], X.shape[2])
    state_to_label = {state: i for i, state in enumerate(set(states))}
    Y = np.array([state_to_label[state] for state in states])
    # PCA with 2 components
    X -= np.mean(X, axis=0)
    U, S, V = np.linalg.svd(X)
    points = np.dot(X, V.T[:,:2])
    # Plot points for each digit
    for i in range(len(set(states))):
        x, y = points[Y == i,:].T
        plt.plot(x, y, 'o', label=str(i))
    plt.legend()
    plt.title('Deep states')
    plt.show()


if __name__ == '__main__':
    # Initialize the weights
    w1 = 0.1 * np.random.randn(nb_hidden_neurons + len(letters), nb_hidden_neurons)
    w2 = 0.1 * np.random.randn(nb_hidden_neurons, len(letters))
    # Learn the weights
    train_reber(w1, w2, 2000)
    # Interpretation
    display_output(w1, w2)
    plot_deep_states(w1, w2)