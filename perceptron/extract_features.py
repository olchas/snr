import glob
import os
import sys

import cv2
import numpy as np
from keras.utils import np_utils

# usage: python extract_features.py ${path_to_directory_with_fruits_data}


def prepare_gabor_filters():
    filters = []
    thetas = [np.pi * i / 8 for i in range(8)]
    lambdas = np.arange(6.0, 11.0)
    sigmas = np.arange(1.0, 6.0)
    for theta in thetas:
        for lambd in lambdas:
            for sigma in sigmas:
                filters.append(cv2.getGaborKernel((9, 9), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F))
    return filters


def extract_gabor_features(img):
    features_vector = []
    for gabor_filter in gabor_filters:
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, gabor_filter)
        # calculate features separately for each RGB channel
        for i in range(3):
            # mean energy
            features_vector.append(np.mean(filtered_img[:, :, i].astype('float64')**2) / 65025)
            # mean_amplitude
            features_vector.append(np.mean(filtered_img[:, :, i]) / 255)
    return features_vector

gabor_filters = prepare_gabor_filters()

print("Reading training data")
training_images = []
training_label = []

for dir_path in glob.glob(os.path.join(sys.argv[1], "Training/*")):
    image_label = dir_path.split("/")[-1]
    iter = 0
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        training_images.append(extract_gabor_features(image))
        training_label.append(image_label)
        iter += 1

training_images = np.array(training_images)
training_label = np.array(training_label)

label_to_id = {v: k for k, v in enumerate(np.unique(training_label))}
id_to_label = {v: k for k, v in label_to_id.items()}

training_label_id = np.array([label_to_id[i] for i in training_label])

print("Reading test data")
test_images = []
test_label = []

for dir_path in glob.glob(os.path.join(sys.argv[1], "Test/*")):
    image_label = dir_path.split("/")[-1]
    iter = 0
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        test_images.append(extract_gabor_features(image))
        test_label.append(image_label)
        iter += 1

test_images = np.array(test_images)
test_label = np.array(test_label)

test_label_id = np.array([label_to_id[i] for i in test_label])

X_train, X_test = training_images, test_images
y_train, y_test = training_label_id, test_label_id

number_of_classes = len(np.unique(y_train))
print("Number of classes: {}".format(number_of_classes))

y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)

if not os.path.exists('../processed_data'):
    os.makedirs('../processed_data')

np.save('../processed_data/X_train_all', X_train)
np.save('../processed_data/X_test_all', X_test)
np.save('../processed_data/y_train_all', y_train)
np.save('../processed_data/y_test_all', y_test)

np.save('../processed_data/label_to_id_all', label_to_id)
np.save('../processed_data/id_to_label_all', id_to_label)



