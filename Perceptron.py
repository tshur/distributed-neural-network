import numpy as np
from itertools import count
from math import sqrt
from dataset import *

class UnthresholdedPerceptron:
    def __init__(self, num_features):
        """Initialize by Gaussian distributed initial weights."""
        self.w = np.mat(np.random.normal(0, 1, num_features))

    def train(self, train, validation, learning_rate=0.01, threshold=0.0001):
        """Train the Perceptron with gradient descent with bold driver rate tuning."""

        for step in count():
            # Amount of change in weights
            delta = learning_rate * self.gradient(train) / len(train)

            # Update weights
            last_w = self.w
            last_err = self.evaluate(validation)
            self.w += delta
            err = self.evaluate(validation)

            # Converged if change in error below threshold
            if abs(err - last_err) < threshold:
                break

            # Bold driver method for tuning learning rate
            if err < last_err:
                learning_rate *= 1.05
            else:
                self.w = last_w
                learning_rate *= 0.5

    def gradient(self, samples):
        """Compute the gradient of sum of squared errors (SSE) loss."""
        delta = np.zeros(samples[0].num_features)
        for sample in samples:
            prediction = self.predict(sample.features)
            for i, value in enumerate(sample.features):
                delta[i] += (sample.label - prediction) * value

        return np.mat(delta)

    def predict(self, features):
        """Run the given features through the Perceptron to make a prediction."""
        return float(self.w * np.mat(features).T)

    def sq_error(self, sample):
        """Calculate the square error for a given sample."""
        return (sample.label - self.predict(sample.features)) ** 2

    def evaluate(self, samples):
        """Evaluate a set of samples using RMSE."""
        return sqrt(sum([self.sq_error(sample) for sample in samples]) / len(samples))

class Perceptron(UnthresholdedPerceptron):
    """Simple Thresholded Perceptron (Returns sign of output)."""
    def predict_output(self, features):
        return sgn(self.predict(features))  # -1 if negative; else, 1

    def error(self, sample):
        """Classification error."""
        return 0 if sample.label == self.predict_output(sample.features) else 1

    def evaluate_error(self, samples):
        """Raw number of samples classified incorrectly."""
        return sum([self.error(sample) for sample in samples])

def sgn(x):
    if x < 0:
        return -1
    return 1

def main():
    """Runs the Perceptron on a simple data set."""
    # Load the energy data and split into datasets for training, validation, and testing.
    samples = load_data('energy-data.txt')
    train, validation, test = divide_dataset(samples)

    # Create an Unthresholded Perceptron and evaluate it with varying convergence thresholds
    up = UnthresholdedPerceptron(samples[0].num_features)
    for thresh in [1e-4]: # [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        up.train(train, validation, threshold=thresh)
        eval_tr = up.evaluate(train)
        eval_v = up.evaluate(validation)
        eval_t = up.evaluate(test)

        print('Train: {:.3f}\tValidation: {:.3f}\tTest: {:.3f}'
              .format(eval_tr, eval_v, eval_t))

if __name__ == '__main__':
    main()