import numpy as np
from node import ParameterNode

class OptimizationAlgorithm:
    def __init__(self, parameter_nodes, learning_rate):
        self.parameter_nodes = parameter_nodes

        # Parameters
        self.learning_rate = learning_rate

    def optimize(self, batch_size=1):
        for i, node in enumerate(self.parameter_nodes):
            direction = self.compute_direction(i, node.get_gradient(0) / batch_size)
            node.w -= self.learning_rate * direction

    def compute_direction(self, i, grad):
        raise NotImplementedError()

class GradientDescent(OptimizationAlgorithm):
    def __init__(self, parameter_nodes, learning_rate):
        OptimizationAlgorithm.__init__(self, parameter_nodes, learning_rate)

    def compute_direction(self, i, grad):
        return grad