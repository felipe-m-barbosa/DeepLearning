import numpy as np
import cv2, sys, math, struct, colorsys, os, argparse, shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder",
                    help="path to folder with images to be converted.", default=None)
parser.add_argument("--results_folder",
                    help="folder path to save the results of convertion.", default=None)
args = parser.parse_args()


if (args.image_folder is None):
    print("Error: missing path_to_convert argument.")
else:
    if (args.results_folder is None):
        print("Warning: no results_path argument informed... Automatically creating folder 'results_HSV' to store the converted images.")
        if (os.path.exists("results_HSV")):
            shutil.rmtree("results_HSV")
        
        os.mkdir("results_HSV")
        path_to_results = os.path.join(os.getcwd(), "results_HSV")
    else:
        path_to_results = os.path.join(os.getcwd(), args.results_folder)
    
    image_folder = args.image_folder
    path_to_convert = os.path.join(os.getcwd(), image_folder)
    for path, subdirs, files in os.walk(path_to_convert):
        for f in tqdm(files, desc='Converting images...'):
            
            im = cv2.imread(os.path.join(path_to_convert, f))
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            #HSV convertion

            for y in range(0,im.shape[0]) :
                for x in range(0,im.shape[1]) :
                    h = im[y, x][0]
                    l = im[y, x][1]
                    if ((h*256 + l) > 0) :
                                    d = 512.0/(h*256 + l)
                    else: d = 0.0
                    color = colorsys.hsv_to_rgb(d,1,1)
                    im[y, x] = [color[0]*255, color[1]*255, color[2]*255]

                    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            #Saving the results
            cv2.imwrite(os.path.join(path_to_results, (f.split('.')[0] + '.png')),im)
