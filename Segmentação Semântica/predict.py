import tensorflow as tf
import numpy as np, cv2

#Personal modules
from utils import dataset_utils, execution_utils, visualization_utils
from models.custom.custom_metrics import mean_iou

def predict_model(model, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE=None, checkpoint_path=None):
    #
    dataset_path = dataset_utils.extract_dataset()

    #
    _, test_filenames = dataset_utils.get_train_test_filenames(dataset_path)

    #
    _, _, test_labels, test_lbl_names = dataset_utils.get_labels(None, test_filenames, one_hot=True)

    #
    _, _, test_images, test_names = dataset_utils.get_input_images(None, test_filenames)

    #
    if model.name == 'Pix2Pix_Gen':
        _, _, test_images, test_labels = dataset_utils.resize_imgs(None, None, test_images, test_labels, height=256, width=256)
    else:
        _, _, test_images, test_labels = dataset_utils.resize_imgs(None, None, test_images, test_labels)

    _, test_images = dataset_utils.rescale_imgs(None, test_images)

    #
    test_images = np.array(test_images)

    if not(checkpoint_path is None):
        # Restoring the latest checkpoint
        latest_cp = tf.train.latest_checkpoint(checkpoint_path)

        # Evaluating the training by loading the previously saved weights in a different instance of the model

        # Loading weigths
        try:
            model.load_weights(latest_cp)
        except FileNotFoundError:
            print("Verify the folder under path: ", checkpoint_path)

        if 'Pix2Pix' in checkpoint_path:
            print("to do...")
        else:
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', mean_iou])

    prediction_batch = model.predict(test_images)

    #visualization_utils.plot_predictions(test_images, test_labels_one_hot, prediction_batch, model.name)

    execution_utils.save_predictions(model.name, test_names, prediction_batch, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)