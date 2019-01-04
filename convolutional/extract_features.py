from numpy.random import seed
seed(1)

import argparse
import os

import h5py
import numpy as np
from keras import Model
from keras.layers import GlobalAveragePooling2D

from h5py_sequence import H5PySequence
from vgg16_models import load_vgg16_model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="path to input directory with preprocessed data")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="path to directory to store extracted features")

    return parser.parse_args()


def build_model():
    # load model without top layers
    vgg16_model = load_vgg16_model(False)

    x = vgg16_model.layers[-1].output

    # add GlobalAveragePooling2D atop of it
    x = GlobalAveragePooling2D()(x)

    return Model(inputs=vgg16_model.input, outputs=x)


def get_target_indexes(seq):
    targets = []
    for i in range(seq.__len__()):
        batch_y = seq.__getitem__(i)[1]
        for one_hot_vector in batch_y:
            targets.append(str(list(one_hot_vector).index(1.0)))

    return targets


def extract_features(features_model, seq, dataset, output_dir):

    features_database = h5py.File(os.path.join(output_dir, dataset), 'w')

    targets = get_target_indexes(seq)

    for class_id in np.unique(targets):
        features_database.create_group(class_id)

    features = features_model.predict_generator(seq)

    for i, class_id in enumerate(targets):

        image_features = features[i]
        features_database[class_id].create_dataset(str(i), data=image_features)

    features_database.close()


def main():
    args = parse_arguments()
    number_of_classes = 83
    batch_size = 16

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = build_model()

    for dataset in ['val', 'test', 'train', 'augmented']:
        seq = H5PySequence(os.path.join(args.input_dir, dataset), number_of_classes, batch_size)
        extract_features(model, seq, dataset, args.output_dir)
        seq.close()

    return 0


if __name__ == "__main__":
    main()