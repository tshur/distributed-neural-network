from collections import deque
import socket
import struct

from mnist_data.load_mnist import load_mnist
from dataset import Sample, load_data, divide_dataset
from NeuralNetwork import FeedForwardNeuralNetwork
from Utils import save_pkl, load_pkl, recv_all
from settings import init

# Gather shared information and data
nn, num_weights, SERVER_INFO, validation = init('validator')
bytes_expected = num_weights * 4

def init_nn():
    """Send initialization string to the server to request to be a validator."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(SERVER_INFO)
        sock.sendall(b'v')  # Ask for initial weights

        # Receive and unpack weights into a list
        data = recv_all(sock, bytes_expected)

    # Unpack weights from bytes and put into a queue for efficient network updating
    init_weights = deque(struct.unpack('{}f'.format(num_weights), data))
    nn.set_weights(init_weights)

def connect_to_master():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(SERVER_INFO)
        data = recv_all(sock, bytes_expected)

    if not data:
        return False

    try:
        # Gather new network weights
        new_network_weights = deque(struct.unpack('{}f'.format(num_weights), data))
    except:
        print('Error unpacking weights!')
        return True

    # Set local network weights
    nn.set_weights(new_network_weights)
    return True

if __name__ == '__main__':
    init_nn()

    still_working = True
    while still_working:
        print('Calculating validation...')
        err = nn.evaluate(validation)
        print('Next mini-batch updated. Validation loss: {:.4f}'.format(err))
        still_working = connect_to_master()

    print('Finished working')