from datetime import datetime
import pickle

class ProgressBar:
    def __init__(self, fill_symbol='#', num_bars=20):
        self.fill_symbol = fill_symbol
        self.num_bars = num_bars
        self.percent = 0  # As decimal (0.20 is 20%)
        self.start_time = None
        self.time_remaining = None

    def refresh(self, new_percent):
        self.update(new_percent)
        self.print()

    def update(self, new_percent):
        self.percent = new_percent
        if not self.start_time:
            self.start_time = datetime.now()
        elif new_percent > 0.0:
            elapsed = datetime.now() - self.start_time
            self.time_remaining = elapsed * ((1.0 - new_percent) / new_percent)

    def clear(self):
        print(' ' * (40 + self.num_bars) + '\r', end='')

    def stamp(self):
        print()

    def print(self):
        num_filled = int(self.percent * self.num_bars)
        num_empty = self.num_bars - num_filled

        if self.time_remaining:
            time_left_str = 'Time Remaining: {}'.format(str(self.time_remaining)[:7])
        else:
            time_left_str = ''

        print(
            ' [{}] {:.2f}% {}\r'
            .format(
                self.fill_symbol * num_filled + ' ' * num_empty,
                self.percent * 100,
                time_left_str),
            end=''
        )

def flatten_image(image):
    """Flatten a 2D image into a single list of pixel values."""
    return [pixel for row in image for pixel in row]

def onehot(value, length):
    """Convert a single value into a onehot vector of given length.

    A onehot vector is a vector with all values LOW except one value
    which is HIGH (i.e., a vector with _one_ _hot_ value).

    Uses 0.1 for the LOW value and 0.9 for the HIGH value (instead of
    0.0 and 1.0) to regularize network training and reduce weight
    magnitudes.

    For example:
    >>> onehot(value=3, length=10)  # e.g., convert digit 3 to onehot
    [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    """
    vector = [0.1] * length
    vector[int(value)] = 0.9
    return vector

def load_pkl(filename):
    """Loads an object from a pickled file."""

    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def save_pkl(obj, filename):
    """Saves an object to a pickled file."""

    with open(filename, 'wb+') as fp:
        pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)

def recv_all(sock, bytes_expected):
    """Receives an expected number of bytes from a socket.

    Continuously RECVs from the socket until expected bytes is reached
    or 0 bytes are received for many consecutive tries. Does NOT
    timeout if the server is busy!
    """
    print('\tReceiving {} bytes...'.format(bytes_expected))

    bytes_received = 0
    tries = 0
    data = bytes()
    while bytes_received < bytes_expected:
        recvd = sock.recv(1024)  # Received as bytes object
        if len(recvd) == 0:
            tries += 1
            if tries > 5:  # Stop if many tries with empty data
                break
        else:
            tries = 0
        # print('\tRECVD: {}'.format(len(recvd)))
        data += recvd
        bytes_received += len(recvd)

    print('\tFinished receiving {} bytes.'.format(bytes_received))
    return data  # bytes object