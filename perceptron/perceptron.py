#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def base_model(functions_set):
    model = Sequential()
    #model.add(Dense(256, activation=activation_function, input_shape=(X_flat_train.shape[1],), kernel_initializer=gabor_weights))
    model.add(Dense(512, activation=functions_set[0], input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation=functions_set[1]))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation=functions_set[2]))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation=functions_set[3]))
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

number_of_classes = len(y_test[0]) # bo tyle jest różnych rodzajów owoców w zbiorze danych -> 81
print("LICZBA KLAS: {}".format(number_of_classes))

# Definicja early stopping z patience=2, zatrzyma nam uczenie sieci, w przypadku, gdy dwie kolejne epoki nie przyniosą
# zysku w postaci zwiększonej dokładności uczenia
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

activation_functions = [p for p in itertools.product(['relu', 'tanh', 'softmax', 'sigmoid'], repeat=4)]

#activation_functions = [p for p in itertools.product(['relu'], repeat=4)]

if not os.path.isdir('models'):
    os.makedirs('models')

score_file = open(os.path.join('models', 'scores.tsv'), 'w')

score_file.write('model\tloss\ttop_1_accuracy\ttop_5_accuracy\n')

for functions_set in activation_functions:

    functions_set_name = '_'.join(functions_set)

    print("WCZYTUJE SIEC!")
    model = base_model(functions_set)
    model.summary()

    output_dir = os.path.join('models', functions_set_name)

    history = model.fit(X_train, y_train, batch_size=64, epochs=200, validation_data=(X_val, y_val), shuffle=True, verbose=1, callbacks=[early_stopping])

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    #history = model.fit(X_flat_train, y_train, batch_size=8, epochs=4, shuffle=True, verbose=1, callbacks=[early_stopping])

    # Wyświetla wykresy uczenia dla funkcji straty (loss) oraz dokładności klasyfikacji
    # list all data in history
    print(history.history.keys())

    # summarize history for top_1_accuracy
    plt.plot(history.history['top_1_accuracy'])
    plt.plot(history.history['val_top_1_accuracy'])
    plt.title('top_1_accuracy')
    plt.ylabel('top_1_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'top_1_accuracy.png'))
    plt.clf()

    # summarize history for top_5_accuracy
    plt.plot(history.history['top_5_accuracy'])
    plt.plot(history.history['val_top_5_accuracy'])
    plt.title('top_5_accuracy')
    plt.ylabel('top_5_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'top_5_accuracy.png'))
    plt.clf()

    prediction = model.predict(X_test)

    score = model.evaluate(X_test, y_test)

    model.save(os.path.join(output_dir, 'model.h5'))

    with open(os.path.join(output_dir, 'history_dict.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    with open(os.path.join(output_dir, 'evaluation.tsv'), 'w') as f:
        f.write('loss\ttop_1_accuracy\ttop_5_accuracy\n')
        f.write('\t'.join(map(lambda x: str(x), score)))

    score_file.write(functions_set_name + '\t' + '\t'.join(map(lambda x: str(x), score)) + '\n')

    with open(os.path.join(output_dir, 'prediction.tsv'), 'w') as f:
        for test_case in prediction:
            f.write('\t'.join(map(lambda x: str(x), test_case)) + '\n')

