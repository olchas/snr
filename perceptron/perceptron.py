#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import matplotlib.pyplot as plt
import numpy as np
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


# Definicja early stopping z patience=2, zatrzyma nam uczenie sieci, w przypadku, gdy dwie kolejne epoki nie przyniosą
# zysku w postaci zwiększonej dokładności uczenia
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

X_train = np.load('../processed_data/X_train_all.npy')
X_test = np.load('../processed_data/X_test_all.npy')
y_train = np.load('../processed_data/y_train_all.npy')
y_test = np.load('../processed_data/y_test_all.npy')

label_to_id = np.load('../processed_data/label_to_id_all.npy')
id_to_label = np.load('../processed_data/id_to_label_all.npy')

number_of_classes = len(y_test[0]) # bo tyle jest różnych rodzajów owoców w zbiorze danych -> 81
print("LICZBA KLAS: {}".format(number_of_classes))


def base_model(activation_function='relu'):
    model = Sequential()
    #model.add(Dense(256, activation=activation_function, input_shape=(X_flat_train.shape[1],), kernel_initializer=gabor_weights))
    model.add(Dense(256, activation=activation_function, input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation=activation_function))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation=activation_function))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation=activation_function))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation=activation_function))
    model.add(Dropout(0.25))
    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[top_1_accuracy, top_5_accuracy])
    return model

print("WCZYTUJE SIEC!")
cnn_n = base_model()
cnn_n.summary()

history = cnn_n.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), shuffle=True, verbose=1, callbacks=[early_stopping])

#history = cnn_n.fit(X_flat_train, y_train, batch_size=8, epochs=4, shuffle=True, verbose=1, callbacks=[early_stopping])

# Wyświetla wykresy uczenia dla funkcji straty (loss) oraz dokładności klasyfikacji
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['top_1_accuracy'])
plt.plot(history.history['val_top_1_accuracy'])
plt.title('top_1_accuracy')
plt.ylabel('top_1_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('top_1_accuracy.png')
# summarize history for loss
plt.plot(history.history['top_5_accuracy'])
plt.plot(history.history['val_top_5_accuracy'])
plt.title('top_5_accuracy')
plt.ylabel('top_5_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('top_5_accuracy.png')

print (cnn_n.predict(X_test))

score = cnn_n.evaluate(X_test, y_test, batch_size=32)

print(score)

