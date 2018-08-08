from operator import itemgetter
from math import sqrt
from collections import deque
from random import choice

from utils import ProgressBar, onehot, flatten_image
from unit import ActivatedUnit, MaxPoolingUnit
from activation import LinearActivation, SigmoidActivation, TanhActivation, ReLUActivation
from layer import Layer, INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER

class FeedForwardNeuralNetwork:
    """A simple FeedForwardNeuralNetwork. Essentially, a FFNN consists of a list of connected Layers.

    Most of the methods simply pass on information to a corresponding method
    for each of the Layers to handle, which in turn pass the information down
    further to their Units to handle.
    """
    def __init__(self, in_features, num_targets, layer_sizes=[5, 10, 5], continuous=False):
        self.continuous = continuous  # CONTINUOUS DATA IS WORK IN PROGRESS -- DOES NOT WORK
        self.num_targets = num_targets
        self.saved_error = None  # Used during training

        hidden_activation = ReLUActivation  # Chosen activation function for Activated units

        # Build input layer
        self.layers = [Layer(ActivatedUnit, in_features - 1, layer_sizes[0], hidden_activation,INPUT_LAYER)]

        # Build hidden layers
        for layer_size in layer_sizes[1:]:
            self.layers.append(Layer(ActivatedUnit, self.layers[-1], layer_size, hidden_activation, HIDDEN_LAYER))

        # Build output layers
        if continuous:
            self.layers.append(Layer(ActivatedUnit, self.layers[-1], num_targets, LinearActivation, OUTPUT_LAYER))
        else:
            self.layers.append(Layer(ActivatedUnit, self.layers[-1], num_targets, hidden_activation, OUTPUT_LAYER))

    def train(self, train, validation, num_epochs=None, learning_rate=0.01, threshold=0.001):
        """Train the FFNN with gradient descent. Dynamic stopping on lowest validation error.

        Training runs over the given number of epochs. If None are given, then training runs
        until the threshold (change in validation error) is reached over multiple consecutive
        iterations. This dynamic stopping also occurs if validation error begins to increase.
        When dynamic stopping is used, the network finalizes the best weights found of the
        duration of training.
        """

        num_epochs_iter = num_epochs if num_epochs else 600  # 600 set to max epochs
        dynamic_stopping = False if num_epochs else True  # Dynamically halt if num_epochs unspec.
        retries = 0
        err = self.evaluate(validation)

        progress_bar = ProgressBar()
        for epoch in range(num_epochs_iter):
            last_err = err
            for i in range(len(train)):
                progress_bar.refresh(i / len(train))
                sample = choice(train)  # Randomly sample training data

                # Update weights based on the chosen sample
                self.prepare_network()
                self.propagate_input(sample.features)
                self.propagate_error(sample.label)
                self.update_weights(sample, learning_rate, momentum=0.3)

            progress_bar.refresh(1.0)
            progress_bar.clear()

            # Evaluate validation error
            err = self.evaluate(validation)
            print('Epoch {} validation error: {:.4f}'.format(epoch, err))
            if dynamic_stopping:
                if last_err - err < threshold:
                    if err <= last_err:  # Still improved, but below threshold
                        self.save_network_weights(err)

                    retries += 1
                    if retries >= 100:
                        epochs_ran = epoch
                        break
                else:
                    self.save_network_weights(err)
                    retries = 0
        else:
            epochs_ran = num_epochs_iter  # Loop did not stop early

        if dynamic_stopping:
            self.finalize_network_weights()  # Finalize weights to best validation error

        return epochs_ran

    def train_worker(self, train, learning_rate=0.01, batch_size=64):
        """Get training weight updates as a worker node (distributed training)."""

        network_weight_updates = deque([None])  # None is a sentinel for init
        progress_bar = ProgressBar()
        for i in range(batch_size):
            progress_bar.refresh(i / batch_size)
            sample = choice(train)  # Randomly sample training data

            # Update weights
            self.prepare_network()
            self.propagate_input(sample.features)
            self.propagate_error(sample.label)
            self.get_weight_updates(sample, learning_rate, network_weight_updates)

        progress_bar.refresh(1.0)
        progress_bar.clear()
        print('Computed weight updates for network on batch_size of {}'.format(batch_size))

        return network_weight_updates

    def train_master(self, network_weight_updates):
        """Update the network based on given updates (distributed training)."""

        print('Starting updating')
        self.update_from_stored_changes(network_weight_updates)
        print('Updated weights')

    def prepare_network(self):
        """Prepare the network by clearing cached inputs and error terms."""
        for layer in self.layers:
            layer.prepare()

    def propagate_input(self, features):
        """Feed the input features forward through the network."""
        for layer in self.layers:
            layer.propagate_input(features)

    def propagate_error(self, target):
        """Propagate the error backwards through the network (backpropagation)."""
        if not self.continuous:
            target = onehot(target, self.num_targets)

        for layer in reversed(self.layers):
            layer.propagate_error(target)

    # def calculate_attention(self):
    #     """Propagate attention backwards through the Layer. Experimental."""
    #     for layer in reversed(self.layers):
    #         layer.propagate_attention()

    #     input_attention = [0] * len(features)
    #     for unit in self.layers[0].units:
    #         for i in range(len(features)):
    #             input_attention[i] += abs(unit.attention * unit.w[i])

    #     print(input_attention)
    #     max_attention = max(enumerate(input_attention), key=itemgetter(1))
    #     print('\nMax attention on feature {} with attention of {}%'
    #           .format(max_attention[0], (100 * max_attention[1]) // sum(input_attention)))

    def update_weights(self, sample, learning_rate, momentum=0.3):
        """Update the network weights according to given momentum and learning_rate."""
        for layer in self.layers:
            layer.update_weights(sample, learning_rate, momentum)

    def get_weight_updates(self, sample, learning_rate, network_weight_updates):
        """Gets the calculated weight updates without modifying the actual network.

        Same as `update_weights()` except modifies `network_weight_updates`,
        a deque, instead of the actual network weights.
        """
        for layer in self.layers:
            layer.get_weight_updates(sample, learning_rate, network_weight_updates)

        if network_weight_updates[0] is None:
            network_weight_updates.popleft()

    def update_from_stored_changes(self, network_weight_updates):
        """Updates the network from a deque() of given weight updates (deltas)."""
        for layer in self.layers:
            layer.update_from_stored_changes(network_weight_updates)

        network_weight_updates.append(None)

    def set_weights(self, network_weights):
        """Sets the network weights based on a deque() of given weights.

        This is done in order of Layers from input to output, and for
        each Layer, in order of Units from lowest index to highest.
        """
        for layer in self.layers:
            layer.set_weights(network_weights)

    def get_weights(self):
        """Fills `network_weights` with the current weights.

        This is done in order of Layers from input to output, and for
        each Layer, in order of Units from lowest index to highest.
        """
        network_weights = deque()
        for layer in self.layers:
            layer.get_weights(network_weights)
        return network_weights

    def save_network_weights(self, error):
        """Temporarily stores the current network weights in each Unit.

        This weights will not be modified or used for future training,
        but reserved for later finalization.
        """
        if not self.saved_error or error < self.saved_error:
            self.saved_error = error  # Stores the best updated validation error
            for layer in self.layers:
                layer.save_weights()

    def finalize_network_weights(self):
        """Sets previously saved weights as the current weights of the network.

        These saved weights must have been stored earlier with the
        `save_weights()` method.
        """
        for layer in self.layers:
            layer.finalize_weights()

    def classify(self, features, verbose=False):
        """Classify a sample (predict its label) from given features."""
        self.prepare_network()
        self.propagate_input(features)
        # self.calculate_attention()

        if self.continuous:
            # if verbose:
            #     print('Prediction: {:.2f}'.format(self.layers[-1][0].out))
            # return self.layers[-1][0].out
            raise NotImplementedError
        else:
            outs = [out_unit.out for out_unit in self.layers[-1].units]  # List of output layer outputs
            prediction = max(enumerate(outs), key=lambda o: o[1])[0]  # Biggest output is prediction
            if verbose:
                print('Prediction: {}, '.format(prediction), end='')
                output = '[' + ', '.join(['{:.2f}'.format(out) for out in outs]) + ']'
                print('Output: {}'.format(output))
            return prediction

    def classification_error(self, sample, verbose=False):
        """Returns the classification error for a sample: 0 if correct; 1 if incorrect."""
        if verbose:
            print('Sample: {}, '.format(sample.label), end='')

        return 1 if sample.label != self.classify(sample.features, verbose) else 0

    def eval_classification(self, samples, verbose=False):
        """Returns the classification error as a decimal for a list of samples."""
        sc_error = 0
        progress_bar = ProgressBar()
        for i, sample in enumerate(samples):
            progress_bar.refresh(i / len(samples))
            sc_error += self.classification_error(sample, verbose)
        progress_bar.refresh(1.0)
        progress_bar.clear()

        return sc_error / len(samples)  # Percent of samples classified incorrectly

    def predict(self, features):
        """Run the given features through the Network to make a prediction."""
        self.prepare_network()
        self.propagate_input(features)

        if self.continuous:
            return self.layers[-1].units[0].out
        else:
            return [out_unit.out for out_unit in self.layers[-1].units]

    def sq_error(self, sample):
        """Calculate the square error for a given sample."""
        prediction = self.predict(sample.features)
        if self.continuous:
            return (sample.label - prediction) ** 2
        else:
            target = onehot(sample.label, self.num_targets)
            return sum([(target[i] - prediction[i]) ** 2 for i in range(len(target))])

    def evaluate(self, samples):
        """Evaluate a set of samples using RMSE."""
        ssq_error = 0
        progress_bar = ProgressBar()
        for i, sample in enumerate(samples):
            progress_bar.refresh(i / len(samples))
            ssq_error += self.sq_error(sample)
        progress_bar.refresh(1.0)
        progress_bar.clear()

        return sqrt(ssq_error / len(samples))

    def write_predictions(self, src_filename, dst_filename):
        """Write predictions on samples from a source file to a destination file."""
        with open(src_filename, 'r') as src, open(dst_filename, 'w+') as dst:
            for line in src:
                if not line: continue

                sample = [float(val) for val in line.strip().split(',')]
                prediction = self.classify(sample) + 1  # Classification scaled to [0, 4]

                dst.write(str(prediction) + '\n')