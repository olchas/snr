import numpy as np
import sys

from keras.models import load_model
from keras import metrics


def top_1_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_5_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)
    
    
model = load_model(sys.argv[1], custom_objects={'top_1_accuracy': top_1_accuracy, 'top_5_accuracy': top_5_accuracy})

X_test = np.load('../processed_data/X_test_all.npy')
y_test = np.load('../processed_data/y_test_all.npy')

id_to_label = np.load('../processed_data/id_to_label_all.npy')

id_to_label_dict = eval(str(id_to_label))

y_test_ids = np.argmax(y_test, axis=1)

predicted_ids = np.argmax(prediction, axis=1)

accuracy = np.sum(predicted_ids == y_test_ids) / len(predicted_ids)

y_test_labels = [id_to_label_dict[x] for x in y_test_ids]

print (accuracy)

