import os
import shutil
import cv2
import time
from tqdm import tqdm
import numpy as np

#Setting up the directory structure
if os.path.exists("data/annots"):
    shutil.rmtree("data/annots")

os.mkdir("data/annots")

dataset_annots_path = "data/annots"

path_to_colored_segmented_images = 'annotated/'

#Creating colors and class_names dicts
ref_img = cv2.imread('annotated/102.png')

ref_img = ref_img/255.0

colors = []

for x in range(ref_img.shape[0]):
    for y in range(ref_img.shape[1]):
        str_val = str(ref_img[x,y])
        if not(str_val in colors):
            colors.append(str_val)

color_values = []
for color in colors:
    channel_values = []
    for channel in color[1:-1].split(' '):
        if (channel != ''):
            channel_values.append(round(float(channel)))

    color_values.append(channel_values)


print(color_values)

color_dict = {idx: color for idx, color in enumerate(color_values)}
              
'''class_names_dict = {0: "Background",
                    1: "Safe Zone",
                    2: "Border", 
                    3: "Object",
                    4: "Unsafe Zone"}
'''

# function to return a dict key given a value
def get_key(my_dict, val):
    for key, value in my_dict.items():
        if np.array_equal(val, value):
             return key

#Iterating over the colored segmented images
for path, subdirs, files in os.walk(path_to_colored_segmented_images):
    for f in tqdm(files, desc="Converting ..."):

        img = cv2.imread(os.path.join(path_to_colored_segmented_images, f))

        img = img/255.0
		
        new_img = np.zeros(img.shape).astype('uint8')

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                color_values_img = [round(channel) for channel in img[x,y]]
                try:
                    new_img[x,y] = get_key(color_dict, color_values_img)
                except:
                    print("Valor que deu pau: ", img[x,y])

        cv2.imwrite(os.path.join(dataset_annots_path, f), new_img)
    
