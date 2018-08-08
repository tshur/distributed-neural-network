from collections import deque
import socket
import struct

from mnist_data.load_mnist import load_mnist
from data.dataset import Sample, load_data, divide_dataset
from model.neural_network import FeedForwardNeuralNetwork
from utils import save_pkl, load_pkl, recv_all
from distributed.settings import init

# Load shared information and data
nn, num_weights, SERVER_INFO, learning_rate, batch_size, train = init('worker')
bytes_expected = num_weights * 4

def init_nn():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(SERVER_INFO)
        sock.sendall(b'w')  # Ask for initial weights as worker node

        # Receive and unpack weights into a list
        data = recv_all(sock, bytes_expected)

    # Unpack and set network weights
    init_weights = deque(struct.unpack('{}f'.format(num_weights), data))
    nn.set_weights(init_weights)

def transmit_to_master(network_weight_updates):
    print('Transmitting updates to master...')  # As packed byte data
    data = struct.pack('{}f'.format(num_weights), *network_weight_updates)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(SERVER_INFO)
        sock.sendall(data)  # Send weight updates to master

        data = recv_all(sock, bytes_expected)  # Receive global network weights

    if not data:
        return False

    try:
        new_network_weights = deque(struct.unpack('{}f'.format(num_weights), data))
    except:
        print('Error unpacking weights!')
        return True

    nn.set_weights(new_network_weights)
    return True

if __name__ == '__main__':
    init_nn()

    still_working = True
    while still_working:
        # Train for a batch with current network weights and return updates to server
        network_weight_updates = nn.train_worker(train, learning_rate=learning_rate, batch_size=batch_size)
        still_working = transmit_to_master(network_weight_updates)

    print('Finished working')