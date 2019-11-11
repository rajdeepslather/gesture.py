# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:14:04 2019

@author: Mohammad
"""

from DT import DataSet

from skimage import io
from skimage.filters import gaussian
from skimage.transform import resize

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


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
            for j in range(1, 2):  # 201):
                label.split()
                image = io.imread('data/leapGestRecog/0' +
                                  str(i) + '/' +
                                  label + '/frame_0' +
                                  str(i) + '_' +
                                  label[0:2] + '_' +
                                  str(j).zfill(4) + '.png')
                # TODO: should we flatten or resize here?
                # img_blurred = gaussian(image, sigma=1.65)
                dataset.inputs.append(resize(image, (60, 160), preserve_range=True))
                dataset.targets.append(int(label[0:2]))

    # initializing the CNN
    classifier = Sequential()

    # Convolution
    classifier.add(Conv2D(32, 3, input_shape=(60, 160, 1), activation="relu", data_format='channels_last'))

    # Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    classifier.add(Flatten())

    classifier.add(Dense(1, activation='relu'))

    # model = Sequential()
    # model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
    # model.add(Activation('softmax'))
    #
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='mean_squared_error', optimizer=sgd)

    classifier.compile(loss='mean_squared_error', optimizer='sgd')

    classifier.fit(np.array(dataset.inputs).reshape(100, 60, 160, 1), dataset.targets)
