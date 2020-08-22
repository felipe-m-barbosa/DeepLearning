import os, shutil, cv2, numpy as np
from tqdm import tqdm

PATH = 'imgs_rotuladas/'

TARGET_PATH = 'imgs_rotuladas_new_colors/'

arquivos = os.listdir(PATH)


if os.path.exists(TARGET_PATH):
	shutil.rmtree(TARGET_PATH)

os.mkdir(TARGET_PATH)

for img_name in tqdm(arquivos, desc='Alterando cores dos arquivos em '+PATH+'...'):
	img = cv2.imread(PATH+img_name)

	for lin in range(img.shape[0]):
		for col in range(img.shape[1]):
			if img[lin][col][0] == 128 and img[lin][col][1] == 0 and img[lin][col][2] == 0:
				img[lin][col] = [34,139,34]
			elif img[lin][col][0] == 0 and img[lin][col][1] == 128 and img[lin][col][2] == 0:
				img[lin][col] = [0,0,255]
			elif img[lin][col][0] == 0 and img[lin][col][1] == 128 and img[lin][col][2] == 128:
				img[lin][col] = [0,255,255]
			elif img[lin][col][0] == 0 and img[lin][col][1] == 0 and img[lin][col][2] == 128:
				img[lin][col] = [255,0,0]


	cv2.imwrite(TARGET_PATH+img_name, img)


'''from matplotlib import pyplot as plt
import numpy as np
import math


def func(pct, allvalues): 
    absolute = math.ceil(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

labels = ['Treinamento', 'Validação', 'Teste']
valores = [415, 47, 100]

plt.title("Distribuição da Base de Dados")
plt.pie(valores, labels=labels, startangle=90, autopct=lambda pct: func(pct, valores),
                                  textprops=dict(color="w", weight='bold'))

plt.legend(loc=4, bbox_to_anchor=(0.8, 0.1, 0.5, 0.5))

plt.show()'''