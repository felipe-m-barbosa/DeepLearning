import cv2
import os
import shutil
import random
import argparse
import time
import sys
from tqdm import tqdm
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('dir_to_split', help='Folder with the images to be splitted')
parser.add_argument('model', help='Model on which the dataset will be used. Possible choices: [vgg, segnet, pix2pix]')
parser.add_argument('-t', '--target_dir', default=None)
args = parser.parse_args()

SOURCE_DIR = args.dir_to_split
MODEL = args.model

assert os.path.exists(SOURCE_DIR), "SOURCE DIR does not exist."

if not(args.target_dir == None):
    TARGET_DIR = args.target_dir
else:
    TARGET_DIR = "dataset_" + MODEL

#assert not(os.path.exists(TARGET_DIR)), "Target DIR already exists"

if (os.path.exists(TARGET_DIR)):
    if strtobool(input("Target DIR already exists. Do you want to overwrite it? [y,n]: ")):
        shutil.rmtree(TARGET_DIR)
    else:
        sys.exit(0)

print("Creating folder under name %s to store the splitted dataset" % (TARGET_DIR))
os.mkdir(TARGET_DIR)

if (MODEL == "vgg" or MODEL == "segnet"):
    #Creating directory tree
    os.makedirs(os.path.join(TARGET_DIR, "train/train-hsv"))
    os.makedirs(os.path.join(TARGET_DIR, "test/test-hsv"))
    os.makedirs(os.path.join(TARGET_DIR, "train/train-orig"))
    os.makedirs(os.path.join(TARGET_DIR, "test/test-orig"))
    os.makedirs(os.path.join(TARGET_DIR, "train/train-labels-int"))
    os.makedirs(os.path.join(TARGET_DIR, "test/test-labels-int"))
    os.makedirs(os.path.join(TARGET_DIR, "train/train-labels"))
    os.makedirs(os.path.join(TARGET_DIR, "test/test-labels"))

    root_hsv_to_split = os.path.join(SOURCE_DIR, "HSV")
    root_orig_to_split = os.path.join(SOURCE_DIR, "original")
    root_int_to_split = os.path.join(SOURCE_DIR, "inteiros")
    root_lbls_to_split = os.path.join(SOURCE_DIR, "rotuladas")

    root_test_hsv = os.path.join(TARGET_DIR, "test/test-hsv")
    root_train_hsv = os.path.join(TARGET_DIR, "train/train-hsv")
    root_test_orig = os.path.join(TARGET_DIR, "test/test-orig")
    root_train_orig = os.path.join(TARGET_DIR, "train/train-orig")
    root_test_lbls_int = os.path.join(TARGET_DIR, "test/test-labels-int")
    root_train_lbls_int = os.path.join(TARGET_DIR, "train/train-labels-int")
    root_test_lbls = os.path.join(TARGET_DIR, "test/test-labels")
    root_train_lbls = os.path.join(TARGET_DIR, "train/train-labels")

    if os.path.exists(root_hsv_to_split) and os.path.exists(root_lbls_to_split):

        num_files = len(os.listdir(root_lbls_to_split))
        print(num_files)


        sorteados = random.sample(range(num_files), num_files-100)

        train_files = []
        test_files = []

        for path, subdirs, files in os.walk(root_hsv_to_split):
            for idx, f in tqdm(enumerate(files), desc="Splitting dataset ... "):
                if idx in sorteados:
                  shutil.copy(os.path.join(path, f), root_train_hsv)
                  shutil.copy(os.path.join(root_orig_to_split, f), root_train_orig)
                  shutil.copy(os.path.join(root_int_to_split, f), root_train_lbls_int)
                  shutil.copy(os.path.join(root_lbls_to_split, f), root_train_lbls)
                  train_files.append(f)
                else:
                    shutil.copy(os.path.join(root_hsv_to_split, f), root_test_hsv)
                    shutil.copy(os.path.join(root_orig_to_split, f), root_test_orig)
                    shutil.copy(os.path.join(root_int_to_split, f), root_test_lbls_int)
                    shutil.copy(os.path.join(root_lbls_to_split, f), root_test_lbls)
                    test_files.append(f)

        with open(os.path.join(TARGET_DIR, 'train_files.txt'), 'w') as f:
            for item in train_files:
                f.write("%s\n" % item)

        with open(os.path.join(TARGET_DIR, 'test_files.txt'), 'w') as f:
            for item in test_files:
                f.write("%s\n" % item)

elif (MODEL == "pix2pix"):
    # Creating directory tree
    os.makedirs(os.path.join(TARGET_DIR, "train/"))
    os.makedirs(os.path.join(TARGET_DIR, "test/"))

    root_test = os.path.join(TARGET_DIR, "test/")
    root_train = os.path.join(TARGET_DIR, "train/")

    num_files = len(os.listdir(SOURCE_DIR))

    sorteados = random.sample(range(num_files), 100)

    train_files = []
    test_files = []

    for path, subdirs, files in os.walk(SOURCE_DIR):
        for idx, f in tqdm(enumerate(files), desc="Splitting dataset ... "):
            if idx in sorteados:
              shutil.copy(os.path.join(path, f), root_test)
              test_files.append(f)
            else:
                shutil.copy(os.path.join(path, f), root_train)
                train_files.append(f)

    with open(os.path.join(TARGET_DIR, 'train_files.txt'), 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)

    with open(os.path.join(TARGET_DIR, 'test_files.txt'), 'w') as f:
        for item in test_files:
            f.write("%s\n" % item)
else:
    if not(os.path.exists(root_imgs_to_split)):
        print("Directory %s doesn't exist.", root_imgs_to_split)
    if not(os.path.exists(root_lbls_to_split)):
        print("Directory %s doesn't exist.", root_lbls_to_split)


'''def confirm():
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = raw_input("OK to push to continue [Y/N]? ").lower()
    return answer == "y"'''
