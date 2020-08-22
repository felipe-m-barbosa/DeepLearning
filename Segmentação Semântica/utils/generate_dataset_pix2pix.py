import cv2, os, shutil, random, argparse, time, numpy as np
from tqdm import tqdm
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('imgs_dir', help='Folder path to the input images')
parser.add_argument('lbls_dir', help='Folder path to the annotations')
parser.add_argument('-t', '--target_dir', help='Folder in which the results will be stored', default=None)
args = parser.parse_args()

IMAGES_PATH = args.imgs_dir
LABELS_PATH = args.lbls_dir

if not (args.target_dir == None):
    TARGET_DIR = args.target_dir
else:
    TARGET_DIR = "results"
    print("The dataset will be stored under folder 'results', as it was not informed by the user")

if (os.path.exists(TARGET_DIR)):
    if strtobool(input("Target DIR already exists. Do you want to overwrite it? [y,n]: ")):
        shutil.rmtree(TARGET_DIR)
    else:
        sys.exit(0)

print("Creating folder under name %s to store the splitted dataset" % (TARGET_DIR))
os.mkdir(TARGET_DIR)

#assert not(os.path.exists(TARGET_DIR)), "Error: target_dir already exists..."

'''print("Creating folder 'results' to store the dataset")
os.mkdir(TARGET_DIR)'''

for path, subdirs, files in os.walk(IMAGES_PATH):
    for f in tqdm(files, desc="Creating dataset and saving it under folder '%s'" % (TARGET_DIR)):
        img = cv2.imread(os.path.join(IMAGES_PATH, f))
        label = cv2.imread(os.path.join(LABELS_PATH, f.split('.')[0]+".png"))

        img = cv2.resize(img, (256,256))
        label = cv2.resize(label, (256,256))

        concat_img = np.concatenate((img,label), axis=1)

        cv2.imwrite(os.path.join(TARGET_DIR, f.split('.')[0] + ".png"), concat_img)