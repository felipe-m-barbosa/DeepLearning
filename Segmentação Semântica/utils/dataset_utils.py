import os, sys, cv2
from zipfile import ZipFile
import numpy as np
from matplotlib import pyplot as plt

#Personal modules
import config

plt.style.use('ggplot')


def extract_dataset(dataset_zip_name='dataset.zip'):
    """
    Extracts the dataset with the same name, and in the same folder, of the zip file.
    :param dataset_zip_name: the name of zip file with the dataset to be used in the training/test phases.
    :return: the localization of the extracted folder
    """
    if not (os.path.exists(dataset_zip_name[:-4])):
        if (os.path.exists(dataset_zip_name)):
            with ZipFile(dataset_zip_name, 'r') as zipObj:
                zipObj.extractall()
        else:
            print("Error: " + dataset_zip_name + " does not exist. Please, upload it before executing script.")
            sys.exit(0)


    return dataset_zip_name[:-4]


def get_train_test_filenames(dataset_path='dataset'):
    """
    Get the train and test filenames and returns it in 2 arrays.
    Expects the filenames stored in txt files and a directory tree like the following:
        dataset
            ...
            train_files.txt
            test_files.txt
    :param dataset_path: dataset location
    :return: the train and test filenames
    """
    train_set_filenames_path = dataset_path + '/train_files.txt'
    test_set_filenames_path = dataset_path + '/test_files.txt'

    train_filenames = []
    test_filenames = []

    train_file = open(train_set_filenames_path, 'r')

    for idx, filename in enumerate(train_file.readlines()):
        train_filenames.append(filename.split('.')[0])

    test_file = open(test_set_filenames_path, 'r')

    for idx, filename in enumerate(test_file.readlines()):
        test_filenames.append(filename.split('.')[0])

    dataset_path = os.path.join(os.getcwd(), "dataset")

    print("Number of train samples: ", len(train_filenames))
    print("Number of test samples: ", len(test_filenames))

    return train_filenames, test_filenames


def show_dataset_content(filenames, num_samples, train=True):
    """
    Plots #num_samples examples of the dataset.
    Each line of the plot contains the Input image, the Ground Truth (annotations) and its colored version.
    :param filenames: a list with the names of the files to be plotted.
    :param num_samples: the number of samples to be plotted.
    :return: the image with #num_samples examples.
    """
    fig, axs = plt.subplots(num_samples, 3, figsize=(10, 10))

    for i, file_name in enumerate(filenames[:num_samples]):
        if (train):
            img = cv2.imread("dataset/train/train-imgs/" + file_name + ".png")
            lbl = cv2.imread("dataset/train/train-labels/" + file_name + ".png")
        else:
            img = cv2.imread("dataset/test/test-imgs/" + file_name + ".png")
            lbl = cv2.imread("dataset/test/test-labels/" + file_name + ".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # As the labels have the same value in the 3 channels, it is not necessary convert the color scheme
        img = cv2.resize(img, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), interpolation=cv2.INTER_NEAREST)
        lbl = cv2.resize(lbl, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), interpolation=cv2.INTER_NEAREST)

        color_labels = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS))
        for x in range(config.IMAGE_HEIGHT):
            for y in range(config.IMAGE_WIDTH):
                color_labels[x, y] = config.COLORS[lbl[x, y, 0]]

        axs[i][0].imshow(img)
        axs[i][0].set_title('Input')
        axs[i][0].axis('off')
        axs[i][1].imshow(lbl)
        axs[i][1].set_title('Label map')
        axs[i][1].axis('off')
        axs[i][2].imshow(color_labels)
        axs[i][2].set_title('Colored labels')
        axs[i][2].axis('off')

    fig.suptitle("Examples of training data", fontsize=14)
    plt.show()

    print("max label value: ", np.max(lbl))
    print("min label value: ", np.min(lbl))

def get_labels(train_filenames, test_filenames, one_hot=False):
    """
    Scorlls through the list of train and test filenames, recovering its correspondent labeled images (annotations)
    and translating them to one-hot representation. Finally, appends the converted images to the lists "train_labels_one_hot"
    and "test_labels_one_hot", which are the return of the function.
    Obs: the expected directory tree is as follows:
        dataset/
            train/
                train-color-labels/
                train-images/
                train-labels/
            test/
                test-color-labels/
                test-images/
                test-labels/
    :param train_filenames: a list with the names of the train files.
    :param test_filenames: a list with the names of the test files.
    :return: the lists with the train/test ground-truth annotations, converted to one-hot representation.
    """

    train_labels = []
    test_labels = []

    train_lbl_names = []
    test_lbl_names = []

    if not(train_filenames is None):
        for i, file_name in enumerate(train_filenames):
            if (one_hot):
                lbl = cv2.imread("dataset/train/train-labels-int/" + file_name + ".png")
            else:
                lbl = cv2.imread("dataset/train/train-labels/" + file_name + ".png")
                
            if (one_hot):
                lbl = np.eye(5)[lbl[:, :, 0]]

            # train_labels_one_hot.append(np.reshape(one_hot_labels, (IMAGE_HEIGHT*IMAGE_WIDTH, -1)))
            train_labels.append(lbl)
            train_lbl_names.append(file_name + ".png")

    if not(test_filenames is None):
        for i, file_name in enumerate(test_filenames):
            if (one_hot):
                lbl = cv2.imread("dataset/test/test-labels-int/" + file_name + ".png")
            else:
                lbl = cv2.imread("dataset/test/test-labels/" + file_name + ".png")

            if(one_hot):
                lbl = np.eye(5)[lbl[:, :, 0]]

            # test_labels_one_hot.append(np.reshape(one_hot_labels, (IMAGE_HEIGHT*IMAGE_WIDTH, -1)))
            test_labels.append(lbl)
            test_lbl_names.append(file_name + ".png")


    return train_labels, train_lbl_names, test_labels, test_lbl_names



def get_input_images(train_filenames, test_filenames, type='hsv'):
    """
    Scorlls through the list of train and test filenames, recovering its correspondent images (inputs for the model)
    Obs: the expected directory tree is as follows:
        dataset/
            train/
                train-images/
                train-labels/
            test/
                test-images/
                test-labels/
    :param train_filenames: a list with the names of the train files.
    :param test_filenames: a list with the names of the test files.
    :return: the lists with the train/test input images.
    """
    train_images = []
    test_images = []

    train_names = []
    test_names = []

    if not(train_filenames is None):
        for i, file_name in enumerate(train_filenames):
            if type=='hsv':
                img = cv2.imread("dataset/train/train-hsv/" + file_name + ".png")
            elif type=='orig':
                img = cv2.imread("dataset/train/train-orig/" + file_name + ".png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # train_labels_one_hot.append(np.reshape(one_hot_labels, (IMAGE_HEIGHT*IMAGE_WIDTH, -1)))
            train_images.append(img)
            train_names.append(file_name + ".png")

    if not(test_filenames is None):
        for i, file_name in enumerate(test_filenames):
            if type=='hsv':
                img = cv2.imread("dataset/test/test-hsv/" + file_name + ".png")
            elif type=='orig':
                img = cv2.imread("dataset/test/test-orig/" + file_name + ".png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # train_labels_one_hot.append(np.reshape(one_hot_labels, (IMAGE_HEIGHT*IMAGE_WIDTH, -1)))
            test_images.append(img)
            test_names.append(file_name + ".png")

    return train_images, train_names, test_images, test_names

def resize_imgs(train_images, train_labels, test_images, test_labels, height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH):
    """
    Resize the images and labels received as parameters to the size (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS).
    :param train_images: list or numpy.ndarray with the images to be used as input in the training phase.
    :param train_labels: list or numpy.ndarray with the labeled images (one-hot) to be used as input in the training
    phase.
    :param test_images: list or numpy.ndarray with the images to be used as input in the testing phase.
    :param test_labels: list or numpy.ndarray with the labeled images (one-hot) to be used as input in the testing
    phase.
    :return: lists with the resized images corresponding to each of the received lists of images.
    """
    train_imgs = []
    train_lbls = []
    test_imgs = []
    test_lbls = []

    if not(train_images is None):
        for train_img in train_images:
            train_img = cv2.resize(train_img, (height, width), interpolation=cv2.INTER_NEAREST)
            train_imgs.append(train_img)
    if not(train_labels is None):
        for train_label in train_labels:
            train_label = cv2.resize(train_label, (height, width), interpolation=cv2.INTER_NEAREST)
            train_lbls.append(train_label)
    if not(test_images is None):
        for test_img in test_images:
            test_img = cv2.resize(test_img, (height, width), interpolation=cv2.INTER_NEAREST)
            test_imgs.append(test_img)
    if not(test_labels is None):
        for test_label in test_labels:
            test_label = cv2.resize(test_label, (height, width), interpolation=cv2.INTER_NEAREST)
            test_lbls.append(test_label)

    return train_imgs, train_lbls, test_imgs, test_lbls

def rescale_imgs(train_images, test_images):
    """
    Rescale the images received as parameters (to be used as the inputs of the model) by a factor of 1/255.0 (float).
    :param train_images: list or numpy.ndarray with the images to be used as input in the training phase.
    :param test_images: list or numpy.ndarray with the images to be used as input in the testing phase.
    :return: lists with the rescaled images corresponding to each of the received lists of images.
    """
    train_imgs = []
    test_imgs = []

    if not (train_images is None):
        for train_img in train_images:
            train_img = train_img/255.0
            train_imgs.append(train_img)

    if not (test_images is None):
        for test_img in test_images:
            test_img = test_img/255.0
            test_imgs.append(test_img)

    return train_imgs, test_imgs