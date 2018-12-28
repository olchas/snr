from numpy.random import seed
seed(1)

import argparse
import glob
import os
import pickle

import cv2
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as image_manip


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="path to input directory where fruits data is stored")
    parser.add_argument("-o", "--output_dir", type=str, default='../convolutional_data',
                        help="path to directory to store output h5py databases")
    parser.add_argument("-r", "--resize_dim", type=int, default=224,
                        help="resized image size")                        

    return parser.parse_args()


def load_images_paths(directory_path):
    images_paths = []
    labels = []
    for dir_path in glob.glob(directory_path):
        image_label = dir_path.split("/")[-1]
        for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
            images_paths.append(image_path)
            labels.append(image_label)
    return np.array(images_paths), np.array(labels)


def load_image(image_path, resize_dim):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # change default opencv BGR channels to RGB to get intended behaviour of vgg16 preprocessing function
    # (they will be switched back)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_preprocessed = preprocess_image(image, resize_dim)
    return image_preprocessed


def preprocess_image(image, resize_dim):
    """Resize image to size required by VGG16 net and apply the same preprocessing as in original net"""
    image_preprocessed = image if resize_dim == 100 else cv2.resize(image, (resize_dim, resize_dim))
    image_preprocessed = image_manip.img_to_array(image_preprocessed)
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
    image_preprocessed = preprocess_input(image_preprocessed)
    return image_preprocessed[0]


def create_augmented_images(image, image_name):
    horizontal_flip = cv2.flip(image, 0)

    height, width = image.shape[:2]
    zoom_factor = 0.8
    zoom_matrix = cv2.getRotationMatrix2D((height / 2, width / 2), 0, zoom_factor)
    zoomed_out = cv2.warpAffine(image, zoom_matrix, (height, width), borderValue=(255, 255, 255))

    # still zoomed out to avoid cropping image in the process
    zoom_factor = 0.7
    rotation_45_matrix = cv2.getRotationMatrix2D((height / 2, width / 2), 45, zoom_factor)
    rotated_45 = cv2.warpAffine(image, rotation_45_matrix, (height, width), borderValue=(255, 255, 255))

    return [horizontal_flip, zoomed_out, rotated_45], [image_name + '_flipped', image_name + '_zoomed_out', image_name + '_rotated']


def main():
    args = parse_arguments()

    # first load just image paths to divide them between train an validation set
    print("Reading training data")
    train_images_paths, train_labels = load_images_paths(os.path.join(args.input_dir, "Training/*"))

    print("Reading test data")
    test_images_paths, test_labels = load_images_paths(os.path.join(args.input_dir, "Test/*"))

    label_to_id = {v: k for k, v in enumerate(np.unique(train_labels))}
    id_to_label = {v: k for k, v in label_to_id.items()}

    train_ids = np.array([label_to_id[i] for i in train_labels])

    test_ids = np.array([label_to_id[i] for i in test_labels])

    # divide training images between train and validation sets
    train_images_paths, val_images_paths, train_ids, val_ids = train_test_split(train_images_paths,
                                                                                train_ids,
                                                                                test_size=0.2)

    dataset_dict = {'train': zip(train_images_paths, map(lambda x: str(x), train_ids)),
                    'val': zip(val_images_paths, map(lambda x: str(x), val_ids)),
                    'test': zip(test_images_paths, map(lambda x: str(x), test_ids))}

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'label_to_id.pkl'), 'wb') as f:
        pickle.dump(label_to_id, f)

    with open(os.path.join(args.output_dir, 'id_to_label.pkl'), 'wb') as f:
        pickle.dump(id_to_label, f)
        
    ids_str = map(lambda x: str(x), id_to_label.keys())
    h5py_databases = {}
    for dataset_name in dataset_dict.keys():
        h5py_databases['dataset_name'] = h5py.File(os.path.join(args.output_dir, dataset_name), 'w')

        # prepare augmented variants for every train image
        if dataset_name == 'train':
            h5py_augmented_database = h5py.File(os.path.join(args.output_dir, 'augmented'), 'w')
            augment = True
            for image_id in ids_str:
                h5py_databases['dataset_name'].create_group(image_id)
                h5py_augmented_database.create_group(image_id)
        else:
            augment = False
            # create group for every class
            for image_id in ids_str:
                h5py_databases['dataset_name'].create_group(image_id)

        for image_path, image_id in dataset_dict[dataset_name]:

            image = load_image(image_path, args.resize_dim)

            image_name = os.path.splitext(os.path.basename(image_path))[0]

            h5py_databases['dataset_name'][image_id].create_dataset(image_name, data=image)

            if augment:
                # add original image to augmented database as well
                h5py_augmented_database[image_id].create_dataset(image_name, data=image)
                augmented_images, augmented_images_names = create_augmented_images(image, image_name)
                for augmented_image, augmented_image_name in zip(augmented_images, augmented_images_names):
                    h5py_augmented_database[image_id].create_dataset(augmented_image_name, data=augmented_image)

        h5py_databases['dataset_name'].close()
        if augment:
            h5py_augmented_database.close()


if __name__ == "__main__":
    main()
