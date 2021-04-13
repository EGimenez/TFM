# usage:

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import platform
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from utils.FeatureProvider import FeatureProvider
from pathlib import Path


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--index', required=False, default=Path() / '..' / 'data' / 'celeba' / 'index' / 'list_attr_celeba_clean.txt',  help='path index file')
if platform.system() == 'Windows':
    ap.add_argument('-n', '--np_files', required=False, default=Path('E:/NOT_BACKUP/TFM') / 'data' / 'celeba' / 'np', help='path directory containing the np files')
else:
    ap.add_argument('-n', '--np_files', required=False, default=Path() / '..' / 'data' / 'celeba' / 'np', help='path directory containing the np files')
ap.add_argument('-r', '--result', required=False, default=Path() / '..' / 'data' / 'celeba' / 'results' / 'linear_analysis' / 'results.json', help='path where result file will be set')
ap.add_argument('-s', '--sampling', required=False, type=float, help='If sampling, indicate the percentage')
ap.add_argument('-m', '--max_sampling', required=False, type=float, default=5000, help='If sampling, indicate the percentage')

args = vars(ap.parse_args())


def run_perceptron(x, y):
    # Perform feature scaling
    sc = StandardScaler()
    x = sc.fit_transform(x)

    perceptron = Perceptron(random_state=0, shuffle=True, class_weight='balanced', fit_intercept=False, verbose=0)
    perceptron.fit(x, y)
    predicted = perceptron.predict(x)
    cm = confusion_matrix(y, predicted)
    return cm


def run_svc(x, y):
    # Perform feature scaling
    sc = StandardScaler()
    x = sc.fit_transform(x)

    svm = SVC(C=1.0, kernel='linear', random_state=0, verbose=1)
    svm.fit(x, y)
    predicted = svm.predict(x)
    cm = confusion_matrix(y, predicted)
    return cm


def show_matrix(cm, title):
    plt.clf()
    plt.imshow(cm, interpolation='nearest')
    classNames = ['Negative', 'Positive']
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


def eval_features(fp: FeatureProvider, result: Path):
    features = fp.get_features()
    results = dict()

    for f, i in zip(features, range(len(features))):
        print(f + ': ' + str(i+1) + '/' + str(len(features)))
        images_0 = fp.get_np({'Blond_Hair': 0}, sample=0.1, sample_max=args['max_sampling'])
        images_1 = fp.get_np({'Blond_Hair': 1}, sample=0.1, sample_max=args['max_sampling'])

        y_0 = np.zeros((len(images_0), 1))
        y_1 = np.ones((len(images_1), 1))

        x = np.concatenate([[*images_0], [*images_1]])
        y = np.append(y_0, y_1)

        # Run Perceptron
        cm = run_perceptron(x, y)
        # show_matrix(cm, f + ': Confusion Matrix - Entire Data')
        print(cm)

        # Cosine
        images_0 = np.average(images_0, axis=0)
        images_1 = np.average(images_1, axis=0)

        cos_sim = dot(images_0, images_1)/(norm(images_0)*norm(images_1))
        print(cos_sim)

        # Run SVM
        run_svc(x, y)

        results[f] = {'cm': cm.tolist(), 'cos': str(cos_sim)}

    if not result.parent.exists():
        result.parent.mkdir()

    with open(result, 'w') as o:
        json.dump(results, o)


if __name__ == '__main__':
    fp = FeatureProvider(index=args['index'], np_path=args['np_files'])
    eval_features(fp, args['result'])
    print('hola')
