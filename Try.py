import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

from skimage import io
import torch


class DataSet:
    def __init__(self, inputs=None, targets=None):
        self.inputs = [] if inputs is None else inputs
        self.targets = [] if targets is None else targets


digits = datasets.load_digits()
# Join the images and target labels in a list
images_and_labels = list(zip(digits.images, digits.target))

# for every element in the list
for index, (image, label) in enumerate(images_and_labels[:8]):
    # initialize a subplot of 2X4 at the i+1-th position
    plt.subplot(2, 4, index + 1)
    # Display images in all subplots
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    # Add a title to each subplot
    plt.title('Training: ' + str(label))

# Show the plot
if __name__ == '__main__':
    # plt.show()
    # print(torch.cuda.get_device_capability())

    dataset = DataSet()
    print('Loading Training Data...')
    labels = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
              '06_index', '07_ok', '08_palm_moved', '09_c', '10_down', ]

    # TODO: remove testing data first
    for i in range(10):
        for label in labels:
            for j in range(1, 201):
                label.split()
                image = io.imread('data/leapGestRecog/0' +
                                  str(i) + '/' +
                                  label + '/frame_0' +
                                  str(i) + '_' +
                                  label[0:2] + '_' +
                                  str(j).zfill(4) + '.png')
                dataset.inputs.append(image)
                dataset.targets.append(label)
                print(image)
