from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras import metrics
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras.backend as K


if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("tf")


def top_1_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_5_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


def build_perceptron(layer_cnt, size_of_layers, activation):
    model = Sequential()
    for i in range(layer_cnt):
        if i == 0:
            model.add(Dense(layer_size, activation=activation,input_shape=(X_train.shape[1],)))
        else:
            model.add(Dense(layer_size, activation=activation))
        model.add(Dropout(0.25))
    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[top_1_accuracy, top_5_accuracy])
    return model
    

X_train = np.load('../processed_data/X_train_all.npy')
X_test = np.load('../processed_data/X_test_all.npy')
y_train = np.load('../processed_data/y_train_all.npy')
y_test = np.load('../processed_data/y_test_all.npy')

label_to_id = np.load('../processed_data/label_to_id_all.npy')
id_to_label = np.load('../processed_data/id_to_label_all.npy')

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

number_of_classes = len(y_test[0])
print("Number of classes: {}".format(number_of_classes))

early_stopping = EarlyStopping(patience=5, min_delta=0.001)

layer_cnt_start = 1
layer_cnt_stop = 4
layer_cnt_step = 1

layer_size_start = 100
layer_size_stop = 1000
layer_size_step = 100

activation_functions = ['sigmoid', 'tanh', 'relu', 'softmax']

if not os.path.isdir('models'):
    os.makedirs('models')

score_file = open(os.path.join('models', 'scores.tsv'), 'w')

score_file.write('activation\tlayer_cnt\tlayer_size\tloss\ttop_1_accuracy\ttop_5_accuracy\n')

for activation in activation_functions:
    for layer_cnt in range(layer_cnt_start, layer_cnt_stop + 1, layer_cnt_step):
        for layer_size in range(layer_size_start, layer_size_stop + 1, layer_size_step):
            model_name = '{activation}_{layer_cnt}_{layer_size}'.format(
                activation=activation, layer_cnt=layer_cnt, layer_size=layer_size)
                
            model = build_perceptron(layer_cnt, layer_size, activation)

            output_dir = os.path.join('models', model_name)

            history = model.fit(X_train, y_train, batch_size=64, epochs=200, validation_data=(X_val, y_val), shuffle=True, verbose=1, callbacks=[early_stopping])

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            prediction = model.predict(X_test)

            score = model.evaluate(X_test, y_test)

            model.save(os.path.join(output_dir, 'model.h5'))

            with open(os.path.join(output_dir, 'history_dict.pkl'), 'wb') as f:
                pickle.dump(history.history, f)

            with open(os.path.join(output_dir, 'evaluation.tsv'), 'w') as f:
                f.write('loss\ttop_1_accuracy\ttop_5_accuracy\n')
                f.write('\t'.join(map(lambda x: str(x), score)))

            score_file.write(str(activation)+'\t'+str(layer_cnt)+'\t'+str(layer_size)+'\t'+'\t'.join(map(lambda x: str(x), score)) + '\n')

            with open(os.path.join(output_dir, 'prediction.tsv'), 'w') as f:
                for test_case in prediction:
                    f.write('\t'.join(map(lambda x: str(x), test_case)) + '\n')

score_file.close()

