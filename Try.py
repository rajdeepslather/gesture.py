import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import tree

from sklearn.metrics import confusion_matrix, classification_report

from skimage import io
from skimage.transform import resize
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

    for i in range(10):
        for label in labels:
            for j in range(1, 61):  # 201):
                label.split()
                image = io.imread('data/leapGestRecog/0' +
                                  str(i) + '/' +
                                  label + '/frame_0' +
                                  str(i) + '_' +
                                  label[0:2] + '_' +
                                  str(j).zfill(4) + '.png')
                # TODO: should we flatten or resize here?
                dataset.inputs.append(resize(image, (60, 160), preserve_range=True).flatten())
                dataset.targets.append(label)

    # TODO: partition into testing and dev

    print('Model Selection...')
    bst_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, max_features=240)

    print('Training...')
    bst_dt.fit(X=dataset.inputs, y=dataset.targets)

    print('Loading Testing Data...')
    test_data = DataSet()
    for i in range(10):
        for label in labels:
            for j in range(61, 91):  # 1, 201):
                label.split()
                image = io.imread('data/leapGestRecog/0' +
                                  str(i) + '/' +
                                  label + '/frame_0' +
                                  str(i) + '_' +
                                  label[0:2] + '_' +
                                  str(j).zfill(4) + '.png')
                test_data.inputs.append(resize(image, (60, 160), preserve_range=True).flatten())
                test_data.targets.append(label)

    print('Testing...')
    predicted = bst_dt.predict(test_data.inputs)
    print("\nClassification Report")
    print(classification_report(predicted, test_data.targets))
