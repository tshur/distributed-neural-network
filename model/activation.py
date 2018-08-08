from math import exp
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    """Abstract base class (ABC) for an Activation Function."""
    def __init__(self):
        pass

    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def derivative(self, output):
        pass


class LinearActivation(ActivationFunction):
    """Linear Activation function (identity)."""
    def activate(self, x):
        return x

    def derivative(self, output):
        return 1.0


class SigmoidActivation(ActivationFunction):
    """Sigmoid Activation function."""
    def activate(self, x):
        if x < -500: return 0.0  # Round underflow
        if x > 500: return 1.0  # Round overflow
        return 1.0 / (1.0 + exp(-float(x)))

    def derivative(self, output):
        """Derivative expressed in terms of activated output (cached)."""
        return output * (1 - output)


class TanhActivation(ActivationFunction):
    """Tanh Activation function."""
    def activate(self, x):
        if x < -500: return -1.0
        if x > 500: return 1.0
        ex, enx = exp(float(x)), exp(-float(x))  # e^x and e^-x
        return (ex - enx) / (ex + enx)

    def derivative(self, output):
        """Derivative expressed in terms of activated output (cached)."""
        return 1.0 - output ** 2


class ReLUActivation(ActivationFunction):
    """Rectified Linear Unit (ReLU) Activation Function."""
    def activate(self, x):
        return max([0.0, x])

    def derivative(self, output):
        if output > 0:
            return 1.0
        else:
            return 0.01  # With leakage to prevent the unit from getting stuck