import os, shutil, cv2, numpy as np
from tqdm import tqdm

arquivos = os.listdir('imgs_rotuladas_new_colors')

TARGET_PATH = 'imgs_int/'

if os.path.exists(TARGET_PATH):
	shutil.rmtree(TARGET_PATH)

os.mkdir(TARGET_PATH)

for img_name in tqdm(arquivos):
	img = cv2.imread('imgs_rotuladas_new_colors/'+img_name)

	int_img = np.zeros_like(img)
	for lin in range(img.shape[0]):
		for col in range(img.shape[1]):
			if img[lin][col][0] == 34 and img[lin][col][1] == 139 and img[lin][col][2] == 34:
				int_img[lin][col] = [1,1,1] #safe_zone
			elif img[lin][col][0] == 0 and img[lin][col][1] == 255 and img[lin][col][2] == 255:
				int_img[lin][col] = [2,2,2] #unsafe_zone
			elif img[lin][col][0] == 0 and img[lin][col][1] == 0 and img[lin][col][2] == 255:
				int_img[lin][col] = [3,3,3] #obstacle
			elif img[lin][col][0] == 255 and img[lin][col][1] == 0 and img[lin][col][2] == 0:
				int_img[lin][col] = [4,4,4] #border
			else:
				int_img[lin][col] = [0,0,0] #background

	cv2.imwrite(TARGET_PATH+img_name, int_img)