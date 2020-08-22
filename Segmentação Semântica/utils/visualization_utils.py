import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

plt.style.use('ggplot')

import config

def turn_into_rgb(preds, colors=config.COLORS):
    """
    Translates a batch of predicted pixel classes to a batch colored images.
    :param preds: prediction batch. Output from calling model.predict()
    :param colors: array with the colors to be used in the tranlation from predited class to colored pixel
    :return: a batch of colored images
    """
    output_batch = []
    for class_map in preds:
        labels = np.argmax(class_map, axis=-1)
        output = np.zeros((class_map.shape[0], class_map.shape[1], config.CHANNELS))
        for x in range(output.shape[0]):
            for y in range(output.shape[1]):
                output[x,y] = colors[labels[x,y]]

        output_batch.append(output)

    return output_batch

def plot_predictions(test_images, test_labels, prediction_batch, model_name, num_samples=5):
    """

    :param test_images:
    :param test_labels:
    :param prediction_batch:
    :param model_name:
    :return:
    """
    colored_prediction_batch = turn_into_rgb(prediction_batch, colors=config.COLORS)
    colored_labels = turn_into_rgb(test_labels, colors=config.COLORS)

    fig, axs = plt.subplots(num_samples, 3, figsize=(20, 20))
    for idx, (img, gt, pred) in enumerate(zip(test_images[:num_samples], colored_labels[:num_samples], colored_prediction_batch[:num_samples])):
        axs[idx][0].imshow(img)
        axs[idx][0].set_title("Input")
        axs[idx][0].axis('off')
        axs[idx][1].imshow(gt)
        axs[idx][1].set_title("Ground truth")
        axs[idx][1].axis('off')
        axs[idx][2].imshow(pred)
        axs[idx][2].set_title("Predicted")
        axs[idx][2].axis('off')

    fig.suptitle("Prections from model "+ model_name)

    plt.show()

def plot_metrics(executions_metrics_dict):
    """

    :param executions_metrics_dict:
    :return:
    """
    ### Ploting metrics grouped by name (*same model and same number of epochs)
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    diff_epochs = np.unique(executions_metrics_dict['num_epochs'])

    for n_epochs in diff_epochs:

        x_range = range(n_epochs)  # select the number of epochs of that execution

        fig, axs = plt.subplots(4, 3, figsize=(20, 10))
        for subp_idx, (metric_name, metric_values) in enumerate(executions_metrics_dict.items()):  # will pass through all the kinds of metrics and its values

            if not (metric_name in ['model', 'num_epochs', 'batch_size','original_trainable']):  # if it is not a hyperparameter of the model
                if not ('evaluate' in metric_name) and not ('training_time' in metric_name):  # evaluation metrics are plotted in a different way (else)
                    #Plotting the history metrics (metricXepochs) for all the model configurations
                    for idx, metric_val in enumerate(metric_values):  # pass through the execution logs, numbering them by the idx
                        if (executions_metrics_dict['num_epochs'][idx] == n_epochs):
                            lbl = str(executions_metrics_dict['model'][idx]) + \
                                  '|' + str(executions_metrics_dict['num_epochs'][idx]) + \
                                  "|" + str(executions_metrics_dict['batch_size'][idx]) + \
                                  "|" + str(executions_metrics_dict['original_trainable'][idx])
                            axs[subp_idx // 3][subp_idx % 3].plot(x_range, metric_val, label=lbl)
                    axs[subp_idx // 3][subp_idx % 3].set_ylabel(metric_name[8:])
                    axs[subp_idx // 3][subp_idx % 3].set_xlabel('epochs')

        fig.suptitle("All executions' metrics", fontsize=14, color='black')
        plt.show()

def plot_metrics_by_execution(all_executions_metrics):
    ### Ploting metrics grouped by model parameters
    for exec_metrics in all_executions_metrics:
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        for idx, (metric_name, metric_value) in enumerate(exec_metrics.items()):
            if idx > 5:  # print just the 6 first metrics
                break

            axs[idx // 3][idx % 3].plot(range(len(metric_value)), metric_value)
            axs[idx // 3][idx % 3].set_ylabel(metric_name[8:])
            axs[idx // 3][idx % 3].set_xlabel('epochs')
            axs[idx // 3][idx % 3].set_title(metric_name, color='black')

        fig.suptitle("Model name: {} | Num epochs: {} | Batch size: {} | Original trainable: {}".format(exec_metrics['model'], exec_metrics['num_epochs'], exec_metrics['batch_size'], exec_metrics['original_trainable']), fontsize=14, color='black')
        plt.show()

'''                else:
                    x_range = range(len(metric_values))
                    x_ticks = [str(executions_metrics_dict['num_epochs'][idx]) + "|" + str(
                        executions_metrics_dict['batch_size'][idx]) + "|" + str(
                        executions_metrics_dict['original_trainable'][idx]) for idx in range(len(metric_values))]
                    axs[subp_idx // 3][subp_idx % 3].bar(x_range, metric_values, label=lbl, color=style_colors)
                    axs[subp_idx // 3][subp_idx % 3].set_xticklabels(x_ticks)
                    axs[subp_idx // 3][subp_idx % 3].set_ylabel(metric_name[9:])

                axs[subp_idx // 3][subp_idx % 3].legend()
                axs[subp_idx // 3][subp_idx % 3].set_title(metric_name, color='black')'''