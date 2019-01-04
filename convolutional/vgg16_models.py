from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import argparse
import os
import pickle

import numpy as np
from keras import Model, metrics, optimizers, activations
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
import keras.backend as K

from h5py_sequence import H5PySequence


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="path to input directory with preprocessed data")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="path to directory to store model, its history and results")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=5,
                        help="number of training epochs")
    parser.add_argument("-d", "--descriptor_size", type=int, default=256,
                        help="number of nodes of the descriptor")
    parser.add_argument("-a", "--activation_function", type=str, choices=["relu", "tanh", "sigmoid", "elu"], default="relu",
                        help="activation function for convolution layers")     
    parser.add_argument("-c", "--convolution_groups", type=int, default=5,
                        help="number of convolution layer groups")                        
    parser.add_argument("-l", "--loss_function", type=str, choices=["mean_squared_logarithmic_error", "logcosh", "categorical_crossentropy", "mean_squared_error"], default="categorical_crossentropy",
                        help="loss function")                                                
    parser.add_argument("-t", "--task", type=str, choices=['2a', '2b', '2c', '2d'], default='2d',
                        help="task to be done")
    parser.add_argument("--no_early_stopping", action='store_true', default=False,
                        help="true if you do not want early stopping of training")
    parser.add_argument("--no_initial_weights", action='store_true', default=False,
                        help="true if you do not want to initialize weights")

    return parser.parse_args()


def top_1_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_5_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


def load_vgg16_model(include_top, weights='imagenet'):
    model = VGG16(weights=weights, include_top=include_top, input_shape=(224, 224, 3))
    return model


def build_model(number_of_classes, task, descriptor_size, activation_function, convolution_groups, no_initial_weights):

    if task in ['2a', '2b', '2c']:
        vgg16_model = load_vgg16_model(True)
        # remove the softmax layer
        vgg16_model.layers.pop()
        if task == '2a':
            return Model(inputs=vgg16_model.input, outputs=vgg16_model.layers[-1].output)

        # get weights of layer
        second_dense_layer_weights = vgg16_model.layers[-1].get_weights()

        # now pop three more layers (Flatten and two Dense)
        vgg16_model.layers.pop()
        vgg16_model.layers.pop()
        vgg16_model.layers.pop()

        # set convolutional layers as nontrainable
        for layer in vgg16_model.layers:
            layer.trainable = False

        x = vgg16_model.layers[-1].output

        # create descriptor on top of convolutional layers
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        predictions = Dense(number_of_classes, activation='softmax')(x)

        model = Model(inputs=vgg16_model.input, outputs=predictions)

        # set weights
        model.layers[-2].set_weights(second_dense_layer_weights)

        return model
                    
    vgg16_model = load_vgg16_model(False, None) if no_initial_weights else load_vgg16_model(False)
   
    # set convolutional layers as nontrainable
    if activation_function == 'relu' and convolution_groups == 5:
        print("Setting convolutional layers as nontrainable")
        for layer in vgg16_model.layers:
            layer.trainable = False

    # if convolution_groups is lower than 5, then pop layers
    if convolution_groups < 5:
        if convolution_groups < 2:
            layers_to_pop = 15
        else:
            layers_to_pop = (5 - convolution_groups) * 4
        for _ in range(layers_to_pop):
            vgg16_model.layers.pop()

    # if activation_function is other than relu, then change it in convolutional layers
    if activation_function != 'relu':
        # take proper function
        activation_function = eval('activations.' + activation_function)
        # for every layer apart from input
        for layer in vgg16_model.layers[1:]:
            # identifying convolutional layers by strides parameter
            if layer.strides == (1, 1):
                layer.activation = activation_function

    x = vgg16_model.layers[-1].output

    # descriptor and classifier
    x = GlobalAveragePooling2D()(x)
    x = Dense(descriptor_size, activation='relu')(x)
    x = Dense(descriptor_size, activation='relu')(x)
    predictions = Dense(number_of_classes, activation='softmax')(x)

    model = Model(inputs=vgg16_model.input, outputs=predictions)
    print('Model loaded')
    
    return model


def fit_model(model, task, train_seq, val_seq, output_dir, loss_function, epochs, early_stopping):

    if task in ['2b', '2c']:
        loss_function = 'categorical_crossentropy'

    model.compile(loss=loss_function, optimizer='adam', metrics=[top_1_accuracy, top_5_accuracy])

    if early_stopping is None:
        history = model.fit_generator(train_seq, validation_data=val_seq, epochs=epochs, verbose=1)
    else:
        print ("Setting early stopping")
        history = model.fit_generator(train_seq, validation_data=val_seq, epochs=epochs, verbose=1, callbacks=[early_stopping])

    model.save(os.path.join(output_dir, 'model.h5'))

    with open(os.path.join(output_dir, 'history_dict.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    return model


def evaluate_model(model, test_seq, output_dir):
    score = model.evaluate_generator(test_seq)
    # prediction = model.predict_generator(test_seq)

    with open(os.path.join(output_dir, 'evaluation.tsv'), 'w') as f:
        f.write('loss\ttop_1_accuracy\ttop_5_accuracy\n')
        f.write('\t'.join(map(lambda x: str(x), score)))

    # with open(os.path.join(output_dir, 'prediction.tsv'), 'w') as f:
    #     for test_case in prediction:
    #         f.write('\t'.join(map(lambda x: str(x), test_case)) + '\n')


def build_preditcion_model(features_model, number_of_classes, train_seq, val_seq, test_seq, output_dir):

    def prepare_target_matrix(seq):
        target = []
        for i in range(seq.__len__()):
            batch_y = seq.__getitem__(i)[1]
            for one_hot_vector in batch_y:
                target.append(one_hot_vector)

        return np.array(target)

    # prepare target matrices
    y_train = prepare_target_matrix(train_seq)
    y_val = prepare_target_matrix(val_seq)
    y_test = prepare_target_matrix(test_seq)

    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    # extract features with vgg16 net with last layer removed
    train_features = features_model.predict_generator(train_seq)
    np.save(os.path.join(output_dir, 'train_features.npy'), train_features)

    val_features = features_model.predict_generator(val_seq)
    np.save(os.path.join(output_dir, 'val_features.npy'), val_features)

    test_features = features_model.predict_generator(test_seq)
    np.save(os.path.join(output_dir, 'test_features.npy'), test_features)

    features_model.save(os.path.join(output_dir, 'features_model.h5'))

    # clear features model from session
    K.clear_session()

    # one layer prediction model
    prediction_model = Sequential()
    prediction_model.add(Dense(number_of_classes, activation='softmax', input_shape=[4096]))
    prediction_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[top_1_accuracy, top_5_accuracy])

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    history = prediction_model.fit(train_features, y_train, batch_size=64, epochs=100, verbose=1, validation_data=(val_features, y_val), shuffle=True, callbacks=[early_stopping])

    prediction_model.save(os.path.join(output_dir, 'prediction_model.h5'))

    with open(os.path.join(output_dir, 'history_dict.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    score = prediction_model.evaluate(test_features, y_test)

    with open(os.path.join(output_dir, 'evaluation.tsv'), 'w') as f:
        f.write('loss\ttop_1_accuracy\ttop_5_accuracy\n')
        f.write('\t'.join(map(lambda x: str(x), score)))


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

    model = build_model(number_of_classes, args.task, args.descriptor_size, args.activation_function, args.convolution_groups, args.no_initial_weights)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.task == '2a':
        build_preditcion_model(model, number_of_classes, train_seq, val_seq, test_seq, args.output_dir)
    else:
        early_stopping = None if args.no_early_stopping else EarlyStopping(patience=1, min_delta=0.005)
        model = fit_model(model, args.task, train_seq, val_seq, args.output_dir, args.loss_function, args.epochs, early_stopping)
        evaluate_model(model, test_seq, args.output_dir)

    train_seq.close()
    val_seq.close()
    test_seq.close()

    return 0


if __name__ == "__main__":
    main()
