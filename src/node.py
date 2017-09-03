import numpy as np

class Node:
    def __init__(self, parents=None, nb_outputs=1):
        # Parents for each input
        self.set_parents(parents or [])
        # Children for each output
        self.children = [[] for _ in range(nb_outputs)]

        # Memoization
        self.x = None
        self.y = None
        self.dJdx = None
        self.dJdy = None

        # Dirty flags
        self.output_dirty = True
        self.gradient_dirty = True

    def set_parents(self, parents):
        # Fill self.parents
        self.parents = []
        for i_input, (parent, i_parent_output) in enumerate(parents):
            self.parents.append((parent, i_parent_output))
            parent.add_child(self, i_input, i_parent_output)
        
    def add_child(self, child, i_child_input, i_output):
        self.children[i_output].append((child, i_child_input))

    def get_output(self, i_output):
        if self.output_dirty:
            self.x = [parent.get_output(i_parent_output) \
                for (parent, i_parent_output) in self.parents]
            self.y = self.compute_output()
            self.output_dirty = False
        return self.y[i_output]

    def compute_output(self):
        raise NotImplementedError()

    def get_gradient(self, i_input):
        # If there are no children, return zero
        if np.sum(len(children) for children in self.children) == 0:
            return np.zeros(self.x[i_input].shape)
        # Get gradient with respect to the i_inputth input
        if self.gradient_dirty:
            self.dJdy = [np.sum(child.get_gradient(i) \
                for child, i in children) for children in self.children]
            self.dJdx = self.compute_gradient()
            self.gradient_dirty = False
        return self.dJdx[i_input]

    def compute_gradient(self):
        raise NotImplementedError()

    def reset_memoization(self):
        # Reset flags
        self.output_dirty = True
        self.gradient_dirty = True

class InputNode(Node):
    def __init__(self, value=None):
        Node.__init__(self)
        self.value = value

    def set_value(self, value):
        self.value = value

    def get_output(self, i_output):
        return self.value

    def compute_gradient(self):
        return [self.dJdy[0]]

class ParameterNode(Node):
    def __init__(self, w):
        Node.__init__(self)
        self.w = w

    def compute_output(self):
        return [self.w]

    def compute_gradient(self):
        return [self.dJdy[0]]

class GradientNode(Node):
    def __init__(self, parents, value=1):
        Node.__init__(self, parents)
        self.value = value

    def set_value(self, value):
        self.value = value

    def compute_output(self):
        return [self.x[0]]

    def get_gradient(self, i_input):
        return self.value

# Operation nodes

class AddBiasNode(Node):
    def compute_output(self):
        return [np.concatenate((np.ones((self.x[0].shape[0], 1)), self.x[0]), axis=1)]

    def compute_gradient(self):
        return [self.dJdy[0][:,1:]]

class IdentityNode(Node):
    def compute_output(self):
        return [self.x[0]]

    def compute_gradient(self):
        return [self.dJdy[0]]

class SigmoidNode(Node):
    def compute_output(self):
        return [1 / (1 + np.exp(-self.x[0]))]

    def compute_gradient(self):
        return [self.dJdy[0] * (self.y[0]*(1 - self.y[0]))]

class TanhNode(Node):
    def compute_output(self):
        return [np.tanh(self.x[0])]

    def compute_gradient(self):
        return [self.dJdy[0] * (1-np.square(self.y[0]))]

class ReluNode(Node):
    def compute_output(self):
        return [np.maximum(0, self.x[0])]

    def compute_gradient(self):
        return [self.dJdy[0] * (self.x[0] >= 0)]

class SoftmaxNode(Node):
    def compute_output(self):
        exp_x = np.exp(self.x[0])
        sums = np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
        return [(exp_x / sums)]

    def compute_gradient(self):
        dJdx = np.zeros((self.x[0].shape))
        for i in range(dJdx.shape[0]):
            y = np.array([self.y[0][i]])
            dydx = -np.dot(y.T, y) + np.diag(self.y[0][i])
            dJdx[i,:] = np.dot(dydx, self.dJdy[0][i])
        return [dJdx]

class ScalarMultiplicationNode(Node):
    def __init__(self, parents, scalar=1):
        Node.__init__(self, parents)
        self.scalar = scalar

    def compute_output(self):
        return [self.scalar * self.x[0]]

    def compute_gradient(self):
        return [self.scalar * self.dJdy[0]]

class Norm2Node(Node):
    def compute_output(self):
        return [np.sum(np.square(self.x[0]))]

    def compute_gradient(self):
        return [2 * self.x[0] * self.dJdy[0]]

class SelectionNode(Node):
    def __init__(self, parents, start=0, end=0):
        Node.__init__(self, parents)
        self.start = start
        self.end = end

    def compute_output(self):
        return [self.x[0][:,self.start:self.end]]

    def compute_gradient(self):
        gradient = np.zeros(self.x[0].shape)
        gradient[:,self.start:self.end] = self.dJdy[0]
        return [gradient]

class AdditionNode(Node):
    def compute_output(self):
        return [self.x[0] + self.x[1]]

    def compute_gradient(self):
        return [self.dJdy[0], self.dJdy[0]]

class SubstractionNode(Node):
    def compute_output(self):
        return [self.x[0] - self.x[1]]
    
    def compute_gradient(self):
        return [self.dJdy[0], -self.dJdy[0]]

class MultiplicationNode(Node):
    def compute_output(self):
        return [np.dot(self.x[0], self.x[1])]

    def compute_gradient(self):
        return [np.dot(self.dJdy[0], self.x[1].T), np.dot(self.x[0].T, self.dJdy[0])]

# Element wise multiplication
class EWMultiplicationNode(Node):
    def compute_output(self):
        return [self.x[0] * self.x[1]]

    def compute_gradient(self):
        return [self.dJdy[0]*self.x[1], self.dJdy[0]*self.x[0]]

class CategoricalCrossEntropyNode(Node):
    def compute_output(self):
        return [-np.sum(self.x[0]*np.log(self.x[1]))]

    def compute_gradient(self):
        return [-self.dJdy[0]*np.log(self.x[1]), -self.dJdy[0]*(self.x[0]/self.x[1])]

class BinaryCrossEntropyNode(Node):
    def compute_output(self):
        return [-np.sum((self.x[0]*np.log(self.x[1]) + (1-self.x[0])*np.log(1-self.x[1])))]

    def compute_gradient(self):
        return [-self.dJdy[0]*(np.log(self.x[1]/(1-self.x[1]))), \
            -self.dJdy[0]*(self.x[0]/self.x[1]-(1-self.x[0])/(1-self.x[1]))]

class SumNode(Node):
    def compute_output(self):
        return [np.sum(self.x, axis=0)]

    def compute_gradient(self):
        return [self.dJdy[0] for _ in self.parents]

class  ConcatenationNode(Node):
    def compute_output(self):
        return [np.concatenate(self.x, axis=1)]

    def compute_gradient(self):
        result = []
        offset = 0
        for i in range(len(self.x)):
            result.append(self.dJdy[0][:,offset:offset+self.x[i].shape[1]])
            offset += self.x[i].shape[1]
        return result