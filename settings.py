"""Shared settings and configuration for distributed nodes."""

from NeuralNetwork import FeedForwardNeuralNetwork
from Utils import load_pkl

IRIS_TRAIN_PKL = 'basic_data/iris-train.pkl'
IRIS_VALID_PKL = 'basic_data/iris-valid.pkl'
IRIS_TEST_PKL = 'basic_data/iris-test.pkl'
MNIST_TRAIN_PKL = 'mnist_data/mnist-train.pkl'
MNIST_VALID_PKL = 'mnist_data/mnist-valid.pkl'
MNIST_TEST_PKL = 'mnist_data/mnist-test.pkl'

run_config = 'IRIS'  # Select the type of data to use and run

if run_config == 'IRIS':  # Configuration for the IRIS data set
    IRIS = {
        'NUM_FEATURES': 4,
        'NUM_OUTPUTS': 3,
        'LAYERS': [16, 8],
        'NUM_WEIGHTS': 227,
        'LEARNING_RATE': 0.001,
        'BATCH_SIZE': 32,
        'NUM_BATCHES': 4000,
        'TRAIN': load_pkl(IRIS_TRAIN_PKL),
        'VALIDATION': load_pkl(IRIS_VALID_PKL),
        'TEST': load_pkl(IRIS_TEST_PKL)
    }
    run_config = IRIS

elif run_config == 'MNIST':  # Configuration for the MNIST data set
    MNIST = {
        'NUM_FEATURES': 784,
        'NUM_OUTPUTS': 10,
        'LAYERS': [500],
        'NUM_WEIGHTS': 397010,  # Directly from layers
        'LEARNING_RATE': 0.001,
        'BATCH_SIZE': 128,
        'NUM_BATCHES': 450 * 5,  # 5 Epochs
        'TRAIN': load_pkl(MNIST_TRAIN_PKL),
        'VALIDATION': load_pkl(MNIST_VALID_PKL),
        'TEST': load_pkl(MNIST_TEST_PKL)
    }
    MNIST['TRAIN'] = MNIST['TRAIN'] + MNIST['VALIDATION'][:-1000]  # Too much validation data
    MNIST['VALIDATION'] = MNIST['VALIDATION'][-1000:]
    run_config = MNIST

else:
    print('Unsupported run_config {}'.format(run_config))
    exit()

SERVER_INFO = ('linux60814', 11235)  # Server hostname and port

def init(node_type, checkpoint_path=''):
    """Initialize shared information and data for given node_type (e.g., 'worker').

    Optionally, can load a stored model from a checkpoint path to continue training or eval.
    """
    if checkpoint_path:
        nn = load_pkl(checkpoint_path)  # Load stored and pickled model
    else:
        nn = FeedForwardNeuralNetwork(  # Return new FFNN
            run_config['NUM_FEATURES'],
            run_config['NUM_OUTPUTS'],
            run_config['LAYERS'])

    if node_type == 'worker':
        data = run_config['TRAIN']  # Worker needs training data, and training hyper parameters
        return nn, run_config['NUM_WEIGHTS'], SERVER_INFO, run_config['LEARNING_RATE'], run_config['BATCH_SIZE'], data

    elif node_type == 'validator':
        data = run_config['VALIDATION']  # Validator only needs validation data
        return nn, run_config['NUM_WEIGHTS'], SERVER_INFO, data

    elif node_type == 'server':
        data = (run_config['TRAIN'], run_config['VALIDATION'], run_config['TEST'])  # Needs all data
        return nn, run_config['NUM_WEIGHTS'], SERVER_INFO, run_config['NUM_BATCHES'], data

    else:
        print('Error: node type not recognized!')
        exit()

    return None