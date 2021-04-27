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
from utils.GlowProvider import GlowProvider
from pathlib import Path
from multiprocessing import Pool


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--index', required=False, default=Path() / '..' / 'data' / 'celeba' / 'index' / 'list_attr_celeba_clean.txt',  help='path index file')
if platform.system() == 'Windows':
    # ap.add_argument('-n', '--np_files', required=False, default=Path('E:/NOT_BACKUP/TFM') / 'data' / 'celeba' / 'np', help='path directory containing the np files')
    ap.add_argument('-n', '--np_files', required=False, default=Path() / '..' / 'data' / 'celeba' / 'np', help='path directory containing the np files')
else:
    ap.add_argument('-n', '--np_files', required=False, default=Path() / '..' / 'data' / 'celeba' / 'np', help='path directory containing the np files')
ap.add_argument('-r', '--result', required=False, default=Path() / '..' / 'data' / 'celeba' / 'results' / 'linear_analysis_avg' , help='path where result file will be set')
ap.add_argument('-s', '--sampling', required=False, type=float, help='If sampling, indicate the percentage')
ap.add_argument('-m', '--max_sampling', required=False, type=float, default=20, help='If sampling, indicate the percentage')

args = vars(ap.parse_args())


if __name__ == '__main__':

    fp = GlowProvider(index=args['index'], np_path=args['np_files'])
    if False:
        eval_features(fp, args['result'])
    if True:
        result = args['result']
        features = fp.get_features()
        results = dict()

        with Pool(5) as p:
            results = p.map(eval_feature_f, features)

        results = {r['feature_name']: {'cm': r['cm'], 'cos': r['cos']} for r in results}

        if not result.parent.exists():
            result.parent.mkdir()

        with open(result, 'w') as o:
            json.dump(results, o)
    print('hola')
