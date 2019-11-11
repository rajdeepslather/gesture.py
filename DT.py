# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:10:01 2019

@author: Rajdeep
"""

import matplotlib.pyplot as plt

from sklearn import svm, tree, datasets
from sklearn.metrics import confusion_matrix, classification_report

from skimage import io
from skimage.filters import gaussian
from skimage.transform import resize


class DataSet:
    def __init__(self, inputs=None, targets=None):
        self.inputs = [] if inputs is None else inputs
        self.targets = [] if targets is None else targets


if __name__ == '__main__':
    dataset = DataSet()
    print('Loading Training Data...')
    labels = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
              '06_index', '07_ok', '08_palm_moved', '09_c', '10_down', ]

    # TODO: need to shuffle data before partitioning
    # 75% to train
    for i in range(10):
        for label in labels:
            for j in range(1, 151):  # 201):
                label.split()
                image = io.imread('data/leapGestRecog/0' +
                                  str(i) + '/' +
                                  label + '/frame_0' +
                                  str(i) + '_' +
                                  label[0:2] + '_' +
                                  str(j).zfill(4) + '.png')
                # TODO: should we flatten or resize here?
                img_blurred = gaussian(image, sigma=1.65)
                dataset.inputs.append(resize(img_blurred, (60, 160), preserve_range=True).flatten())
                dataset.targets.append(label)

    print('Model Selection...')
    bst_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, max_features=240)

    print('Training...')
    bst_dt.fit(X=dataset.inputs, y=dataset.targets)

    print('Loading Testing Data...')
    # 25% to test
    test_data = DataSet()
    for i in range(10):
        for label in labels:
            for j in range(151, 201):  # 1, 201):
                label.split()
                image = io.imread('data/leapGestRecog/0' +
                                  str(i) + '/' +
                                  label + '/frame_0' +
                                  str(i) + '_' +
                                  label[0:2] + '_' +
                                  str(j).zfill(4) + '.png')
                img_blurred = gaussian(image, sigma=1.65)
                test_data.inputs.append(resize(img_blurred, (60, 160), preserve_range=True).flatten())
                test_data.targets.append(label)

    print('Testing...')
    predicted = bst_dt.predict(test_data.inputs)
    print("\nClassification Report")
    print(classification_report(predicted, test_data.targets))
