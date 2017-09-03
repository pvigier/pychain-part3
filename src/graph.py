from node import *

class Graph:
    def __init__(self, nodes, input_nodes, output_nodes, expected_output_nodes, cost_node, parameter_nodes):
        self.nodes = nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.expected_output_nodes = expected_output_nodes
        self.cost_node = cost_node
        self.parameter_nodes = parameter_nodes

        # Create a gradient node
        GradientNode([(cost_node, 0)])

    def propagate(self, X):
        self.reset_memoization()
        for x, node in zip(X, self.input_nodes):
            node.set_value(x)
        return [node.get_output(0) for node in self.output_nodes]

    def backpropagate(self, Y):
        for y, node in zip(Y, self.expected_output_nodes):
            node.set_value(y)
        cost = self.cost_node.get_output(0)
        for node in self.parameter_nodes:
            node.get_gradient(0)
        return cost

    def reset_memoization(self):
        for node in self.nodes:
            node.reset_memoization()

    def get_parameter_nodes(self):
        return self.parameter_nodes