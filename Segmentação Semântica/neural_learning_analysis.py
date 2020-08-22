import os
from utils.execution_utils import load_training_parameters
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')
style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

keys = ['model', 'batch_size', 'original_trainable', 'history_loss', 'history_accuracy', 'history_mean_iou', 'history_val_loss', 'history_val_accuracy', 'history_val_mean_iou', 'training_time', 'evaluate_loss', 'evaluate_accuracy', 'evaluate_mean_iou']
model_names = ['SegNet', 'FCN_VGG16_32s', 'FCN_VGG16_16s', 'FCN_VGG16_8s', 'DenseNet', 'Pix2Pix']


dict_50_epc = {key : [] for key in keys}

#========================
#Loading metrics
#========================
for path, _, files in os.walk('metrics'):
	for f in files:
		if not(f.endswith('.py')):
			params_dict, params_vect = load_training_parameters(f)
			if 'epc_50_' in f:
				for key in keys:
					dict_50_epc[key].append(params_dict[key][0])
			else:
				print("Ainda precisa fazer")


for model in model_names:
	fig, axs = plt.subplots(3,1,figsize=(10,20))
	for idx, m_name in enumerate(dict_50_epc['model']):
		if m_name == model:
			for idx2, BS in enumerate([2,4,8]):
				print(dict_50_epc['batch_size'][idx])
				if dict_50_epc['batch_size'][idx] == BS:
					if dict_50_epc['original_trainable'] == True:
						axs[idx2].plot(np.arange(50), dict_50_epc['history_mean_iou'][idx], label='Trainable: True')
					else:
						axs[idx2].plot(np.arange(50), dict_50_epc['history_mean_iou'][idx], label='Trainable: False')
				axs[idx2].set_title("batch_size = " + str(BS), size=11)
				axs[idx2].set_ylabel("mean IoU", size = 9)
				axs[idx2].set_xlabel("Epochs", size=9)
				axs[idx2].legend()

	if model == 'DenseNet':
		fig.suptitle("Neural Learning FCN_" + model, size=13)
	else:
		fig.suptitle("Neural Learning " + model, size=13)
	plt.show()

'''eixo_y_mious = []

for idx, mod_name in enumerate(model_names):
	for idx_2, mn in enumerate(dict_50_epc['model']):
		if mn == mod_name and dict_50_epc['batch_size'][idx_2] == 2 and dict_50_epc['original_trainable'][idx_2] == True:
			eixo_y_mious.append(dict_50_epc['evaluate_mean_iou'][idx_2])

eixo_y_mious.append(0.903)

num_params_modelos = [29459225, 134383434, 134309594, 134292079, 58526282, 54419459]

eixo_x = num_params_modelos

max_limit = max(eixo_x)
min_limit = min(eixo_x)


print(eixo_y_mious)


plt.figure(figsize=(10,10))
for idx_2, (x, y) in enumerate(zip(eixo_x, eixo_y_mious)):
  plt.scatter(((x-min_limit)/(max_limit-min_limit))+0.1, y, s=70)

plt.ylim((0,1))
plt.title("Model efficiency", size=18)
plt.tick_params(axis='x', size = 10)
plt.tick_params(axis='y', size = 10)
plt.grid(True)
plt.xlabel("Number of parameters", size=14, weight='bold')
plt.ylabel("mean IoU", size=14, weight='bold')

for idx_2, (x, y) in enumerate(zip(eixo_x, eixo_y_mious)):
	# this method is called for each point
    plt.annotate(model_names[idx_2], # this is the text
                (((x-min_limit)/(max_limit-min_limit))+0.1,y+0.02), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center',
                weight='bold') # horizontal alignment can be left, right or center
  
    plt.annotate('('+str(int(x))+', '+str(round(y, 2))+')', # this is the text
                  (((x-min_limit)/(max_limit-min_limit))+0.1,y), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(0,10), # distance from text to points (x,y)
                  ha='center') # horizontal alignment can be left, right or center

plt.show()'''