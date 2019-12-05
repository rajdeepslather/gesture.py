# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:41:44 2019

@author: Mohammad
"""

from skimage import io
from skimage.filters import gaussian
from skimage.transform import resize
import matplotlib.pyplot as plt

import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


class DataSet:
    def __init__(self, inputs=None, targets=None):
        self.inputs = [] if inputs is None else inputs
        self.targets = [] if targets is None else targets


if __name__ == '__main__':

    dataset = DataSet()
    print('Loading Training Data...')
    labels = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
              '06_index', '07_ok', '08_palm_moved', '09_c', '10_down', ]

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
                img_blurred = gaussian(image, sigma=1.65)
                dataset.inputs.append(resize(img_blurred, (30, 80), preserve_range=True))
                dataset.targets.append(int(label[1]))
    x_data_org = np.array(dataset.inputs, dtype='float32')
    y_data_org = np.array(dataset.targets)
    y_data_org = y_data_org.reshape(150 * 100, 1)  # Reshape to be the correct size

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
                test_data.inputs.append(resize(img_blurred, (30, 80), preserve_range=True))
                test_data.targets.append(int(label[1]))
    x_test_org = np.array(test_data.inputs, dtype='float32')
    y_test_org = np.array(test_data.targets)
    y_test_org = y_test_org.reshape(50 * 100, 1)  # Reshape to be the correct size

    # shuffle data
    x_data, y_data = shuffle(x_data_org, y_data_org)
    x_validate, x_data, y_validate, y_data = train_test_split(x_data, y_data, test_size=0.75)

    # initializing the CNN
    classifier = Sequential()

    # Convolution
    classifier.add(Convolution2D(16, 3, 3, input_shape=(30, 80, 1), activation='relu'))

    # Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # add a second layer
    classifier.add(Convolution2D(32, 3, 3, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    classifier.add(Flatten())

    # densing
    classifier.add(Dense(output_dim=64, activation='relu'))
    classifier.add(Dense(output_dim=10, activation='softmax'))

    # compiling CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    from keras.utils import to_categorical

    y_data = to_categorical(y_data)
    x_data = x_data.reshape((len(x_data), 30, 80, 1))
    #    x_data /= 255
    y_validate = to_categorical(y_validate)
    x_validate = x_validate.reshape((len(x_validate), 30, 80, 1))
    #    x_validate /= 255
    y_test = to_categorical(y_test_org)
    x_test = x_test_org.reshape((len(x_test_org), 30, 80, 1))

    H = classifier.fit(x_data, y_data,
                       epochs=5, batch_size=32, verbose=1,
                       validation_data=(x_validate, y_validate))

    print('Testing...')
    predicted = classifier.predict(x_test)
    predicted_classes = np.argmax(predicted, axis=1)

    print("\nClassification Report")

    print(classification_report(np.argmax(y_test, axis=1), predicted_classes))
    print("\nConfusion Matrix")
    print(confusion_matrix(np.argmax(y_test, axis=1), predicted_classes))

    print(classifier.summary())

    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])
    plt.show()

    plt.plot(H.history['accuracy'])
    plt.plot(H.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'])
    plt.show()
