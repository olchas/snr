from numpy.random import seed
seed(1)

import h5py
import numpy as np
from keras.utils import Sequence
from keras.utils import to_categorical


class H5PySequence(Sequence):

    def __init__(self, hp5_database_path, number_of_classes, batch_size):
        self.database = h5py.File(hp5_database_path, 'r')
        self.number_of_classes = number_of_classes
        self.batch_size = batch_size

        self.database_size = sum([len(self.database[class_id]) for class_id in self.database])
        self.number_of_batches = int(np.ceil(self.database_size / self.batch_size))
        self.array_indexes = np.arange(self.database_size)

        self._setup_data()
        self.on_epoch_end()

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, idx):
        batch_keys = self.image_keys[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.database[class_id][image_name] for (class_id, image_name) in batch_keys]

        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def _setup_data(self):
        self.image_keys = []
        image_ids = []
        for class_id in self.database.keys():
            for image_name in self.database[class_id].keys():
                self.image_keys.append((class_id, image_name))
            image_ids += [int(class_id)] * len(self.database[class_id])
        self.image_keys = np.array(self.image_keys)
        self.y = to_categorical(image_ids, num_classes=self.number_of_classes)

    def on_epoch_end(self):
        np.random.shuffle(self.array_indexes)

        self.image_keys = self.image_keys[self.array_indexes]
        self.y = self.y[self.array_indexes]

    def close(self):
        self.database.close()
