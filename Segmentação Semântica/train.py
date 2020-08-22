import tensorflow as tf

import shutil, os, time, cv2

# Personal modules
from utils import execution_utils
from models.custom.custom_metrics import mean_iou
import config


def fit_model(model, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset, test_images, test_labels_one_hot):
    CHECKPOINT_PATH = 'checkpoints/'+model.name+'/cp-' + ("bs_"+str(BATCH_SIZE)+"_epc_"+str(NUM_EPOCHS)+"_tnb_"+str(config.IS_ORIGINAL_TRAINABLE)) + '/{epoch:04d}.ckpt'
    CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

    #Defining a keras callback to save the weights of the model in a certain frequency
    if not(os.path.exists(CHECKPOINT_DIR)):
        os.makedirs(CHECKPOINT_DIR)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        verbose=1,
        save_weights_only=True,
        period=config.SAVE_PERIOD)

    # when running in your local machine, use 'logs\scalars'
    logdir = os.path.join("logs\\scalars", (
            model.name + "bs_" + str(BATCH_SIZE) + "_epc_" + str(NUM_EPOCHS) + "_tnb_" + str(
        IS_ORIGINAL_TRAINABLE)))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    if model.name == 'AlexNet':
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto',
            min_lr=0.00001
        )

        sgd = tf.keras.optimizers.SGD(
            learning_rate=0.01, momentum=0.9, decay=0.0005
        )

        callbacks_list = [cp_callback, tb_callback, lr_callback]

    else:
        sgd = tf.keras.optimizers.SGD()
        callbacks_list = [tb_callback]

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', mean_iou])

    start_time = time.time()

    history = model.fit(train_dataset,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=len(train_dataset),
                        validation_data=val_dataset,
                        validation_steps=len(val_dataset),
                        callbacks=callbacks_list)

    end_time = time.time()

    training_time = end_time - start_time
    print("Elapsed time: ", training_time)

    execution_utils.save_training_parameters(model, test_images, test_labels_one_hot, training_time, history.history, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)

    return model