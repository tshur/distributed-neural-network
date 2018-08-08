import socketserver
import struct
from collections import deque

from mnist_data.load_mnist import load_mnist
from data.dataset import Sample, load_data, divide_dataset
from model.neural_network import FeedForwardNeuralNetwork
from utils import save_pkl, load_pkl, flatten_image, recv_all
from distributed.settings import init

# Initialize shared information and data
nn, num_weights, SERVER_INFO, total_batches, data = init('server')
train, validation, test = tuple(data)

bytes_expected = num_weights * 4  # Receives weights as 4byte floats over network

batch_no = 0

# Keep track of the set of connected & initialized workers and validators
workers = set()
validators = set()

class RequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        """A client connected to the server; handle the request."""

        global batch_no

        if self.client_address[0] in workers:  # Established worker
            # self.request is a TCP socket connected to the client
            print('Worker connected... handling')
            self.data = recv_all(self.request, bytes_expected)

            try:
                network_weight_updates = deque(struct.unpack('{}f'.format(num_weights), self.data))
                print('\tReceived data unpacked')
            except:
                print('\tError unpacking data')
            else:
                print('\tTraining master network')
                nn.train_master(network_weight_updates)
                batch_no += 1
                print('\tCompleted batch #{} of {}'.format(batch_no, total_batches))

            print('\tGetting network weights')
            network_weights = nn.get_weights()
            print('\tPacking response')
            response = struct.pack('{}f'.format(num_weights), *network_weights)
            print('\tSending back weights')
            self.request.sendall(response)

        elif self.client_address[0] in validators:  # Established validator
            print('Validator connected... handling')
            print('\tGetting network weights')
            network_weights = nn.get_weights()
            print('\tPacking response')
            response = struct.pack('{}f'.format(num_weights), *network_weights)
            print('\tSending back weights')
            self.request.sendall(response)

        else:  # New node connected
            # Initialize client into set of workers
            print('Contacted by new node...')
            self.data = self.request.recv(1024).strip()
            print('\tReceived init string')

            # New nodes should send b'w' to become a worker, or b'v' to become a validator
            if self.data == b'w':
                print('\tNew WORKER node... handling')
                workers.add(self.client_address[0])  # Add IP to list of workers
            elif self.data == b'v':
                print('\tNew VALIDATOR node... handling')
                validators.add(self.client_address[0])  # Add IP to list of validators
            else:
                print('\tUnknown node request! Moving on...')
                return

            # Gather current network weights, pack as bytes, and send back to client
            network_weights = nn.get_weights()
            response = struct.pack('{}f'.format(num_weights), *network_weights)
            print('\tSending network weights')
            self.request.sendall(response)

        print('\tDONE with node')

        if batch_no > total_batches:
            # Done with desired number of batches! Finalize network training
            save_pkl(nn, 'saved_nn/iris_nn_relu_16n8n_100e_16b_l2.pkl')  # Save trained network

            print('Eval training...')
            eval_tr = nn.eval_classification(train)
            print('Eval validation...')
            eval_v = nn.eval_classification(validation)
            print('Eval testing...')
            eval_t = nn.eval_classification(test)

            print('Batches: {}\tTrain: {:.3f}\tValidation: {:.3f}\tTest: {:.3f}'
                  .format(batch_no, eval_tr, eval_v, eval_t))
            exit()

def main():
    socketserver.TCPServer.allow_reuse_address = True  # So IP/PORT will be reused by program
    with socketserver.TCPServer(SERVER_INFO, RequestHandler) as server:
        print('Listening at {} on port {}...'.format(server.server_address, SERVER_INFO[1]))
        server.serve_forever()  # Infinitely serve clients
        if batch_no > total_batches:
            print('Closing server...')
            server.server_close()

if __name__ == '__main__':
    main()
