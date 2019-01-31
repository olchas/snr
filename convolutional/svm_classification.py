from h5py_sequence import H5PySequence
import numpy as np
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import json


def get_dataset(dir, batch_size):
    seq = H5PySequence(dir, 83, batch_size)
    X = [seq.database[class_id][image_name].value for (class_id, image_name) in seq.image_keys]
    y = [np.where(el == 1)[0][0] for el in seq.y]
    return X, y


def init_svc(gamma):
    # Default values
    # kernel = rbf (exponential kernel)
    # gamma = 1/n_features
    svm = OneVsRestClassifier(SVC(kernel="rbf", gamma=gamma, max_iter=1000))
    return svm


if __name__ == "__main__":
    X_train, y_train = get_dataset('train', 16*2140)
    X_test, y_test = get_dataset('test', 1600*9)

    gamma_reports = {}
    preds = {}
    for gamma in [0.001, 0.010, 0.050, 0.100, 1.000, 5.000, 10.000, 100.000, 500.000, 1000.000]:
        svc = init_svc(gamma)
        svc = OneVsRestClassifier(SVC(kernel='rbf', gamma=gamma, max_iter=200))
        svc.fit(X_train, y_train)
        pred = svc.predict(X_test)
        gamma_reports[gamma] = classification_report(pred, y_test, output_dict=True)
        preds[gamma] = pred

    with open("gamma_reports.json", "w") as f:
        json.dump({k:list(v) for k,v in gamma_reports.items()}, f)
    with open("preds.json", "w") as f:
        json.dump(preds, f)