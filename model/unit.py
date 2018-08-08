from math import fabs
from abc import ABC, abstractmethod
import numpy as np

INPUT_LAYER = 0
HIDDEN_LAYER = 1
OUTPUT_LAYER = 2

class Unit(ABC):
    """Abstract Base Class (ABC) for a general network Unit."""
    def prepare(self):
        """Prepares the network, flushing the stored input and error."""
        self.error = None
        self.out = None
        self.attention = None

    def output(self):
        """Computes the Unit output and stores in Unit.out."""
        self.out = self.predict()

        # For Units connected to this Unit's output, transmit the data
        if not self.is_output_unit():
            for output in self.outputs:
                output.x[self.index] = self.out

        return self.out

    @abstractmethod
    def calculate_error(self, target):
        """Calculates and stores the error based on target value and gradient."""
        pass

    def feed_input(self, features):
        """Set given features as inputs to the unit."""
        for i in range(self.dim):
            self.x[i] = features[i]

    def save_weights(self):
        """Save the current weights (separate from training weights) for later use."""
        self.saved_w = self.w[:]

    def finalize_weights(self):
        """Finalize the previously-saved weights as the current training weights."""
        self.w = self.saved_w[:]

    # Returns the type of the Unit
    def is_input_unit(self):
        return self.layer == INPUT_LAYER
    def is_hidden_unit(self):
        return self.layer == HIDDEN_LAYER
    def is_output_unit(self):
        return self.layer == OUTPUT_LAYER

class ActivatedUnit(Unit):
    def __init__(self, index, num_features, outputs, layer, activation):
        """Initialize by Gaussian distributed initial weights."""
        self.index = index
        self.dim = num_features
        self.x = [None] * (num_features - 1) + [1]
        self.delta_prev = [0] * num_features
        self.w = list(np.random.normal(0, 0.05, num_features))
        self.saved_w = self.w[:]
        self.out = None
        self.error = None
        self.outputs = outputs
        self.layer = layer
        self.activation = activation

    def predict(self):
        """Run the given features through the Unit to make a prediction."""
        return self.activation.activate(float(np.mat(self.w) * np.mat(self.x).T))

    def calculate_error(self, target):
        """Calculate the error based on the backpropagation algorithm."""
        if self.is_output_unit():
            error = target[self.index] - self.out
        else:
            error = 0
            for output in self.outputs:
                error += output.error * output.w[self.index]

        self.error = error * self.activation.derivative(self.out)  # Gradient term
        return self.error

    def calculate_attention(self):
        """EXPERIMENTAL attention calculation.

        Distribute attention (set as 1.0 for total attention at the output
        Units) from output Units backwards to the input Units analogously to
        the backpropagation algorithm. Weights are distributed to inputs
        based on the amount of contribution from the input and weights.
        Once completed, the input units should have an attention score
        proportional to the amount of attention, influence, or effect that input
        unit (or feature) had on the final output.
        """
        if self.is_output_unit():
            attention = 1.0
        else:
            attention = 0  # try scaling
            for output in self.outputs:
                attention += fabs(output.w[self.index] * output.attention * self.out)
        self.attention = attention
        return attention


# UNTESTED and IN PROGRESS
class MaxPoolingUnit(Unit):
    def __init__(self, index, output, kernel=[2, 2]):
        """Initialize by Gaussian distributed initial weights."""
        self.index = index
        self.kernel = kernel
        self.x = [[None for _ in self.kernel[1]] for _ in self.kernel[0]]
        self.out = None
        self.error = None
        self.output = output

    def predict(self):
        max_val = None
        for row in range(self.kernel[1]):
            for col in range(self.kernel[0]):
                val = self.x[row][col]
                if not max_val or val > max_val:
                    max_val = val

        return val

    def prepare(self):
        self.error = None
        self.out = None
        for row in range(self.kernel[1]):
            for col in range(self.kernel[0]):
                self.x[row][col] = None

    def feed_input(self, features):
        for row in range(self.kernel[1]):
            for col in range(self.kernel[0]):
                self.x[row][col] = features[row][col]

    def output(self):
        self.out = self.predict()
        output.x[self.index] = self.out
        return self.out

    def calculate_error(self, target):
        self.error = output.error * output.w[self.index]
        return self.error


# UNFINISHED
class ConvolutionalUnit(Unit):
    def __init__(self, index, num_features, outputs, layer, activation):
        """Initialize by Gaussian distributed initial weights."""
        self.index = index
        self.dim = num_features
        self.x = [None] * (num_features - 1) + [1]
        self.delta_prev = [0] * num_features
        self.w = list(np.random.normal(0, 0.05, num_features))
        self.saved_w = self.w[:]
        self.out = None
        self.error = None
        self.outputs = outputs
        self.layer = layer
        self.activation = activation

    def prepare(self):
        self.error = None
        self.out = None
        self.attention = None
        for i in range(self.dim - 1):
            self.x[i] = None

    def predict(self):
        """Run the given features through the Unit to make a prediction."""
        return self.activation.activate(float(np.mat(self.w) * np.mat(self.x).T))

    def feed_input(self, features):
        print(features)
        print(len(features))
        for i in range(self.dim):
            print(i)
            self.x[i] = features[i]

    def save_weights(self):
        self.saved_w = self.w[:]

    def finalize_weights(self):
        self.w = self.saved_w[:]

    def output(self):
        self.out = self.predict()

        if not self.is_output_unit():
            for output in self.outputs:
                output.x[self.index] = self.out

        return self.out

    def calculate_error(self, target):
        if self.is_output_unit():
            error = target[self.index] - self.out
        else:
            error = 0
            for output in self.outputs:
                error += output.error * output.w[self.index]

        self.error = error * self.activation.derivative(self.out)
        return self.error

    def is_input_unit(self):
        return self.layer == INPUT_LAYER
    def is_hidden_unit(self):
        return self.layer == HIDDEN_LAYER
    def is_output_unit(self):
        return self.layer == OUTPUT_LAYER