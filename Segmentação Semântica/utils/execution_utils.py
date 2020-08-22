import json, os, shutil, cv2, numpy as np
import tensorflow as tf

#Personal modules
import config
from utils import visualization_utils
from models.custom.custom_metrics import mean_iou, mean_iou_p2p, acc_p2p
from models.custom.custom_losses import sparse_categ_cross_entropy

def evaluate_p2p(model, test_images, test_labels):
    gen_imgs = model.predict(test_images)

    mean_iou = mean_iou_p2p(test_labels, gen_imgs)
    acc = acc_p2p(test_labels, gen_imgs)
    loss = sparse_categ_cross_entropy(test_labels, gen_imgs)

    metrics = [loss, acc, mean_iou]

    return metrics


def save_training_parameters(model, test_images, test_labels, training_time, training_history, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE=None):
    """

    :param model:
    :param test_images:
    :param test_labels:
    :param training_time:
    :param training_history:
    :return:
    """
    if model.name == 'Pix2Pix_Gen':
        metrics = evaluate_p2p(model, test_images, test_labels)
        model_name = 'Pix2Pix'
    else:
        metrics = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
        model_name = model.name
    #print("model.metrics_names: ", model.metrics_names)

    metrics_dict = {"history_" + key: [float(i) for i in value] for key, value in training_history.items()}

    for metric_name, metric_value in zip(['loss', 'accuracy', 'mean_iou'], metrics):
        metrics_dict["evaluate_" + metric_name] = float(metric_value)

    metrics_dict['training_time'] = float(training_time)
    metrics_dict["original_trainable"] = IS_ORIGINAL_TRAINABLE
    metrics_dict["model"] = model_name
    metrics_dict["num_epochs"] = NUM_EPOCHS
    metrics_dict["batch_size"] = BATCH_SIZE

    if not(os.path.exists('metrics')):
        os.mkdir('metrics')

    if os.path.exists('metrics\\'+model_name + '_metrics_history_bs_'+str(BATCH_SIZE)+'_epc_'+str(NUM_EPOCHS)+'_tnb_'+str(IS_ORIGINAL_TRAINABLE)+'.txt'):
        with open('metrics\\'+model_name + '_metrics_history_bs_'+str(BATCH_SIZE)+'_epc_'+str(NUM_EPOCHS)+'_tnb_'+str(IS_ORIGINAL_TRAINABLE)+'.txt', 'a') as file:
            file.write("\n")
            file.write(json.dumps(metrics_dict))  # use `json.loads` to do the reverse
    else:
        with open('metrics\\'+model_name + '_metrics_history_bs_'+str(BATCH_SIZE)+'_epc_'+str(NUM_EPOCHS)+'_tnb_'+str(IS_ORIGINAL_TRAINABLE)+'.txt', 'w') as file:
            file.write(json.dumps(metrics_dict))  # use `json.loads` to do the reverse


def load_training_parameters(file_name):
    """

    :param model:
    :return:
    """
    with open('metrics/'+file_name, 'r') as handle:
        all_executions_metrics = [json.loads(line) for line in handle.readlines()]

    all_executions_metrics_dict = {key: [] for key in all_executions_metrics[0].keys()}

    for stored_metrics in all_executions_metrics:
        for idx, (metric_name, metric_value) in enumerate(stored_metrics.items()):
            all_executions_metrics_dict[metric_name].append(metric_value)

    return all_executions_metrics_dict, all_executions_metrics



def save_predictions_2(targets, prediction_batch, epoch, bs, results_path):
    """

    :param model:
    :param test_names:
    :param prediction_batch:
    :param NUM_EPOCHS:
    :param BATCH_SIZE:
    :param IS_ORIGINAL_TRAINABLE:
    :return:
    """

    targets = targets.numpy()
    prediction_batch = prediction_batch.numpy()

    targets = [(prediction*0.5 + 0.5)*255 for prediction in targets]
    targets = np.array(targets)
    
    prediction_batch = [(prediction*0.5 + 0.5)*255 for prediction in prediction_batch]

    prediction_batch = np.array(prediction_batch)

    #targets_one_hot = convert_one_hot(targets, bs)
    #prediction_batch_one_hot = convert_one_hot(prediction_batch, bs)
    
    last_idx = 0
    results_path = results_path+'/bs_'+str(bs)+'/ep_'+str(epoch)

    if not(os.path.exists(results_path+'/targets')):
        os.makedirs(results_path+'/targets')
    else:
        arquivos = os.listdir(results_path+'/targets')
        nomes_arquivos = [int(n.split('.')[0]) for n in arquivos]
        last_idx = max(nomes_arquivos)

    if not(os.path.exists(results_path+'/preds')):
        os.makedirs(results_path+'/preds')

    for idx, (targ, pred) in enumerate(zip(targets, prediction_batch)):
        name = str((idx+1)+last_idx) + '.png'
        pred = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2BGR)
        targ = cv2.cvtColor(targ.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(results_path+'/targets', name), targ)
        cv2.imwrite(os.path.join(results_path+'/preds', name), pred)

    print("salvou")

def save_predictions(model_name, test_names, prediction_batch, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE=None):
    """

    :param model:
    :param test_names:
    :param prediction_batch:
    :param NUM_EPOCHS:
    :param BATCH_SIZE:
    :param IS_ORIGINAL_TRAINABLE:
    :return:
    """
    
    colored_prediction_batch = visualization_utils.turn_into_rgb(prediction_batch, colors=config.COLORS)
    results_path = "results/" + model_name + "/" + (
                "bs_" + str(BATCH_SIZE) + "_epc_" + str(NUM_EPOCHS) + "_tnb_" + str(
            IS_ORIGINAL_TRAINABLE))
    if (os.path.exists(results_path)):
        shutil.rmtree(results_path)

    os.makedirs(results_path)
    for name, pred in zip(test_names, colored_prediction_batch):
        pred = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(results_path, name), pred)