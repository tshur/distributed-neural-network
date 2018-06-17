import numpy as np
import cv2
import os
import math
from scipy import ndimage

from Utils import load_pkl
from dataset import Sample
from NeuralNetwork import FeedForwardNeuralNetwork

MNIST_MODEL = 'saved_nn/nn_relu_500n_5e_64b_l3.pkl'

CUSTOM_MNIST_DIRECTORY = 'custom_mnist/class_digits'

def preprocess_handwritten(data_folder):
    """Given a folder containing images, pre-process for MNIST format.

    The format results in a white digit over a black background. Then,
    the image is normalized in [0, 1] and then flattened into a list
    before being returned in a list with the image labels. Input images
    should be stored in data_folder/raw/ and each image filename should
    begin with the label: e.g., 0.jpg or 0_image.jpg.

    Formatting involves centering the digit by center of mass, making
    the background black based on a threshold, and then padding the
    image on all sides with black pixels.
    """
    # based on https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4

    # the folders to contain our data
    raw_filepath = os.path.join(data_folder, 'raw')
    proc_filepath = os.path.join(data_folder, 'proc')  # Destination for processed images

    # List of valid image filenames
    image_filenames = [
        filename for filename in os.listdir(raw_filepath)
        if '.jpg' in filename and filename[0].isdigit()
    ]
    num_images = len(image_filenames)

    # np.arrays that we will fill with out image/label data
    images = np.zeros((num_images, 784), dtype=np.float32)
    labels = np.zeros((num_images, 1), dtype=np.int32)

    # process each digit 0-9 one at a time
    for i, image_name in enumerate(image_filenames):
        label = image_name[0]
        print('FILE: {}, LABEL: {}'.format(image_name, label))

        # load the image as a grayscale
        gray = cv2.imread(os.path.join(raw_filepath, image_name),
                          cv2.IMREAD_GRAYSCALE)

        # resize to 28x28 and invert to white writing on black background
        gray = cv2.resize(255-gray, (28, 28))

        # change gray to black if darker than a threshold
        (thresh, gray) = cv2.threshold(gray, 128, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Begin reformatting to center a 20x20 digit into a 28x28 box
        while np.sum(gray[0]) == 0:
            gray = gray[1:]
        while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1)
        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]
        while np.sum(gray[:,-1]) == 0:
            gray = np.delete(gray,-1,1)

        # handle resizing to 20x20
        rows, cols = gray.shape
        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols*factor))
            gray = cv2.resize(gray, (cols,rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows*factor))
            gray = cv2.resize(gray, (cols, rows))

        # Now pad to add the black edges to make 28x28
        colsPadding = (int(math.ceil((28-cols)/2.0)),
                       int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),
                       int(math.floor((28-rows)/2.0)))
        gray = np.lib.pad(gray,(rowsPadding, colsPadding), 'constant')

        # Now center based on center of mass
        def getBestShift(img):
            cy,cx = ndimage.measurements.center_of_mass(img)

            rows,cols = img.shape
            shiftx = np.round(cols/2.0-cx).astype(int)
            shifty = np.round(rows/2.0-cy).astype(int)

            return shiftx,shifty

        def shift(img,sx,sy):
            rows,cols = img.shape
            M = np.float32([[1,0,sx],[0,1,sy]])
            shifted = cv2.warpAffine(img,M,(cols,rows))  # matrix transform
            return shifted

        # Apply shifting of inner box to be centered based on centerof mass
        shiftx, shifty = getBestShift(gray)
        shifted = shift(gray, shiftx, shifty)
        gray = shifted

        # save processed images
        if not os.path.exists(proc_filepath):
            os.mkdir(proc_filepath)
        cv2.imwrite(os.path.join(proc_filepath, image_name), gray)

        # scale 0 to 1
        flat = gray.flatten() / 255.0

        images[i] = flat
        labels[i] = label

    return images, labels

def main():
    # Gather and preprocess handwritten digits given in CUSTOM_MNIST_DIRECTORY
    images, labels = preprocess_handwritten(CUSTOM_MNIST_DIRECTORY)
    mini_test = [
        Sample(list(image) + list(label))  # Convert to Sample type
        for image, label in zip(images, labels)
    ]

    trained_nn = load_pkl(MNIST_MODEL)  # Load a pre-trained MNIST model
    err = trained_nn.eval_classification(mini_test, verbose=True)
    print(
        'The model got {:.0f} / {} correct! ({:.1f}%)'
        .format((1 - err) * len(mini_test), len(mini_test), 100 * (1 - err))
    )

if __name__ == '__main__':
    main()