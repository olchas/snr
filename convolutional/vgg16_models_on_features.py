from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import argparse
import os

from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
import keras.backend as K

from h5py_sequence import H5PySequence
from vgg16_models import load_vgg16_model, fit_model, evaluate_model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="path to input directory with preprocessed data")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="path to directory to store model, its history and results")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("-d", "--descriptor_size", type=int, default=256,
                        help="number of nodes of the descriptor")
    parser.add_argument("-a", "--activation_function", type=str, choices=["relu", "tanh", "sigmoid", "elu"], default="relu",
                        help="activation function for convolution layers")
    parser.add_argument("-l", "--loss_function", type=str, choices=["mean_squared_logarithmic_error", "logcosh", "categorical_crossentropy", "mean_squared_error"], default="categorical_crossentropy",
                        help="loss function")
    parser.add_argument("-t", "--task", type=str, choices=['2b', '2c', '2d'], default='2d',
                        help="task to be done")
    parser.add_argument("--no_early_stopping", action='store_true', default=False,
                        help="true if you do not want early stopping of training")
                        
    return parser.parse_args()


def build_prediction_model(number_of_classes, task, descriptor_size, activation_function):

    if task in ['2b', '2c']:
        vgg16_model = load_vgg16_model(True)

        # get weights of second Dense layer
        second_dense_layer_weights = vgg16_model.layers[-2].get_weights()

        K.clear_session()

        descriptor_size = 4096
        activation_function = 'relu'

    # create prediction model
    prediction_model = Sequential()
    prediction_model.add(Dense(descriptor_size, activation=activation_function, input_shape=[512]))
    if task in ['2b', '2c']:
        prediction_model.add(Dense(descriptor_size, activation=activation_function, weights=second_dense_layer_weights))
    else:
        prediction_model.add(Dense(descriptor_size, activation=activation_function))
    prediction_model.add(Dense(number_of_classes, activation='softmax'))

    print('Model loaded')

    return prediction_model


def main():
    args = parse_arguments()
    number_of_classes = 83

    if args.task == '2c':
        train_seq = H5PySequence(os.path.join(args.input_dir, 'augmented'), number_of_classes, args.batch_size)
    else:
        train_seq = H5PySequence(os.path.join(args.input_dir, 'train'), number_of_classes, args.batch_size)

    print('Train sequence loaded')

    val_seq = H5PySequence(os.path.join(args.input_dir, 'val'), number_of_classes, args.batch_size)

    print('Validation sequence loaded')

    test_seq = H5PySequence(os.path.join(args.input_dir, 'test'), number_of_classes, args.batch_size)

    print('Test sequence loaded')

    model = build_prediction_model(number_of_classes, args.task, args.descriptor_size, args.activation_function)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    early_stopping = None if args.no_early_stopping else EarlyStopping(patience=1, min_delta=0.005)
    model = fit_model(model, args.task, train_seq, val_seq, args.output_dir, args.loss_function, args.epochs,
                      early_stopping)
    evaluate_model(model, test_seq, args.output_dir)

    train_seq.close()
    val_seq.close()
    test_seq.close()

    return 0


if __name__ == "__main__":
    main()
