from utils import visualization_utils, execution_utils
import numpy as np
import matplotlib.pyplot as plt

segnet_metrics_dict, segnet_metrics = execution_utils.load_training_parameters('Segnet')
vgg32s_metrics_dict, vgg32s_metrics = execution_utils.load_training_parameters('FCN_VGG16_32s')
vgg16s_metrics_dict, vgg16s_metrics = execution_utils.load_training_parameters('FCN_VGG16_16s')
vgg8s_metrics_dict, vgg8s_metrics = execution_utils.load_training_parameters('FCN_VGG16_8s')

diff_epochs = np.unique(segnet_metrics_dict['num_epochs'])

print(diff_epochs)

control = np.zeros(len(segnet_metrics_dict.keys()))

fig, axs = plt.subplots(4, 4, figsize=(20, 10))
'''for n_epochs in diff_epochs:
    x_range = range(n_epochs)  # select the number of epochs of that execution
    for metric_dict in [segnet_metrics_dict, vgg32s_metrics_dict, vgg16s_metrics_dict, vgg8s_metrics_dict]: #will pass through all the dictionaries
        control = True
        for metric_idx, (metric_name, metric_values) in enumerate(metric_dict.items()):  # will pass through all the kinds of metrics and its values
            if not (metric_name in ['model', 'num_epochs', 'batch_size', 'original_trainable']):  # if it is not a hyperparameter of the model
                if ('history' in metric_name):
                    for idx, metric_val in enumerate(metric_values):
                        if (metric_dict['num_epochs'][idx] == n_epochs):
                            lbl = str(metric_dict['model'][idx]) + \
                                  '|' + str(metric_dict['num_epochs'][idx]) + \
                                  "|" + str(metric_dict['batch_size'][idx]) + \
                                  "|" + str(metric_dict['original_trainable'][idx])
                            #acumulating the metrics in a plot
                            axs[plot_control // 3][plot_control % 3].plot(x_range, metric_val, label=lbl)


                            axs[metric_idx // 3][metric_idx % 3].set_title(metric_name[8:]+"|"+str(n_epochs))
                            axs[metric_idx // 3][metric_idx % 3].set_ylabel(metric_name[8:])
                            axs[metric_idx // 3][metric_idx % 3].set_xlabel('epochs')
                            axs[metric_idx // 3][metric_idx % 3].legend()
                            plot_control'''


fig.suptitle("All executions' history metrics", fontsize=14, color='black')
plt.show()