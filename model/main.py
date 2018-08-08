from mnist_data.load_mnist import load_mnist
from data.dataset import Sample, load_data, divide_dataset
from model.neural_network import FeedForwardNeuralNetwork
from utils import flatten_image

IRIS_DATA = 'basic_data/iris-data.txt'

def run_iris():
    """Run IRIS data, a small classification task of different flower species."""
    samples = load_data(IRIS_DATA)
    train, validation, test = divide_dataset(samples)

    # Train the network with the given hyperparameters and evaluate
    nn = FeedForwardNeuralNetwork(samples[0].num_features, 3, [6])
    epochs = nn.train(train, validation, learning_rate=0.01, threshold=1e-5)
    eval_tr = nn.eval_classification(train)
    eval_v = nn.eval_classification(validation)
    eval_t = nn.eval_classification(test)

    print('Epochs: {}\tTrain: {:.3f}\tValidation: {:.3f}\tTest: {:.3f}'
          .format(epochs, eval_tr, eval_v, eval_t))

def run_mnist():
    """Run MNIST data, a large classification task of handwritten digits."""
    train_images, train_labels, test_images, test_labels = load_mnist()
    train = [Sample(flatten_image(image) + [label])
             for image, label in zip(train_images, train_labels)]  # 55,000 images
    test = [Sample(flatten_image(image) + [label])
            for image, label in zip(test_images, test_labels)]  # 10,000 images

    # Split further into train and validation
    train, validation, _ = divide_dataset(train, train_r=0.8, validation_r=0.2, test_r=0.0)

    # Train the network with the given hyperparameters and evaluate
    nn = FeedForwardNeuralNetwork(train[0].num_features, 10, [64])
    epochs = nn.train(train, validation, learning_rate=0.01, threshold=1e-3)
    eval_tr = nn.eval_classification(train)
    eval_v = nn.eval_classification(validation)
    eval_t = nn.eval_classification(test, verbose=True)

    print('Epochs: {}\tTrain: {:.3f}\tValidation: {:.3f}\tTest: {:.3f}'
          .format(epochs, eval_tr, eval_v, eval_t))

def main():
    run_iris()
    # run_mnist()

if __name__ == '__main__':
    main()