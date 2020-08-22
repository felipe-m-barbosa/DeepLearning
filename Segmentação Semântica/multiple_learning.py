import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import count_params

#Personal modules
from models.fcn import FCN_VGG16_32s, FCN_VGG16_16s, FCN_VGG16_8s, DenseNet
from models.segnet import Segnet
from models.alexnet import AlexNet
import config, train, predict
from utils import dataset_utils
from tensorflow.keras.backend import clear_session


#Loading and configuring the dataset
#
dataset_path = dataset_utils.extract_dataset()

#
train_filenames, test_filenames = dataset_utils.get_train_test_filenames(dataset_path)

#
train_labels_one_hot, train_lbl_names, test_labels_one_hot, test_lbl_names = dataset_utils.get_labels(
    train_filenames, test_filenames, one_hot=True)

#
train_images, train_names, test_images, test_names = dataset_utils.get_input_images(train_filenames, test_filenames)

#
train_images, train_labels_one_hot, test_images, test_labels_one_hot = dataset_utils.resize_imgs(train_images,
                                                                                                train_labels_one_hot,
                                                                                                test_images,
                                                                                                test_labels_one_hot)

#
train_images, test_images = dataset_utils.rescale_imgs(train_images, test_images)

# Ordering
'''zipped_train_imgs = zip(train_names, train_images)
train_images = [img for name, img in sorted(zipped_train_imgs)]

zipped_train_lbls = zip(train_lbl_names, train_labels_one_hot)
train_labels_one_hot = [img for name, img in sorted(zipped_train_lbls)]

zipped_test_imgs = zip(test_names, test_images)
test_images = [img for name, img in sorted(zipped_test_imgs)]

zipped_test_lbls = zip(test_lbl_names, test_labels_one_hot)
test_labels_one_hot = [img for name, img in sorted(zipped_test_lbls)]'''

#
train_images = np.array(train_images)
train_labels_one_hot = np.array(train_labels_one_hot)
test_images = np.array(test_images)
test_labels_one_hot = np.array(test_labels_one_hot)

datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.1)

#data_generator = datagen.flow(train_images[0:10], save_to_dir='imagens_aug', save_format='jpeg', save_prefix='aug')

#vgg_32s_tnb_false = FCN_VGG16_32s(tnb_extractor=False)
#vgg_32s = FCN_VGG16_32s()
#vgg_16s_tnb_false = FCN_VGG16_16s(tnb_extractor=False)
#vgg_16s = FCN_VGG16_16s()
#vgg_8s_tbn_false = FCN_VGG16_8s(tnb_extractor=False)
#vgg_8s = FCN_VGG16_8s()
#segnet_tnb_false = Segnet(tnb_extractor=False)
#segnet = Segnet()
dense_net_tnb_false = DenseNet(tnb_extractor=False)
dense_net = DenseNet()



#old_weights_vgg32s = vgg_32s.get_weights()
#old_weights_vgg32s_tnb_false = vgg_32s_tnb_false.get_weights()
#old_weights_vgg16s = vgg_16s.get_weights()
#old_weights_vgg16s_tnb_false = vgg_16s_tnb_false.get_weights()
#old_weights_vgg8s = vgg_8s.get_weights()
#old_weights_vgg8s_tnb_false = vgg_8s_tbn_false.get_weights()
#old_weights_segnet = segnet.get_weights()
#old_weights_segnet_tnb_false = segnet_tnb_false.get_weights()
old_weights_dense_net = dense_net.get_weights()
old_weights_dense_net_tnb_false = dense_net_tnb_false.get_weights()


for BATCH_SIZE in [8]:
    for NUM_EPOCHS in [50]:
        for IS_ORIGINAL_TRAINABLE in [True]:
            train_dataset = datagen.flow(train_images, train_labels_one_hot, batch_size=BATCH_SIZE, subset='training')
            val_dataset = datagen.flow(train_images, train_labels_one_hot, subset='validation')

            print("IS_ORIGINAL_TRAINABLE: ", IS_ORIGINAL_TRAINABLE)
            print("NUM_EPOCHS: ", NUM_EPOCHS)
            print("BATCH_SIZE: ", BATCH_SIZE)

            print("Training densenet")
            if (IS_ORIGINAL_TRAINABLE):
            	# reseting the model
            	dense_net.set_weights(old_weights_dense_net)
            	train.fit_model(dense_net, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
	                                test_images, test_labels_one_hot)
            	predict.predict_model(dense_net, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)
            else:
            	# reseting the model
            	dense_net_tnb_false.set_weights(old_weights_dense_net_tnb_false)
            	train.fit_model(dense_net_tnb_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
                                test_images, test_labels_one_hot)
            	predict.predict_model(dense_net_tnb_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)

            '''print("Training vgg32s")
                                                if (IS_ORIGINAL_TRAINABLE):
                                                    # reseting the model
                                                    vgg_32s.set_weights(old_weights_vgg32s)
                                                    train.fit_model(vgg_32s, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
                                                                    test_images, test_labels_one_hot)
                                                    predict.predict_model(vgg_32s, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)
                                                else:
                                                    vgg_32s_tnb_false.set_weights(old_weights_vgg32s_tnb_false)
                                                    train.fit_model(vgg_32s_tnb_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
                                                                    test_images, test_labels_one_hot)
                                                    predict.predict_model(vgg_32s_tnb_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)'''
                                                
                                                #clear_session()
                                                
            '''print("Training vgg16s")
                                                if (IS_ORIGINAL_TRAINABLE):
                                                    # reseting the model
                                                    vgg_16s.set_weights(old_weights_vgg16s)
                                                    train.fit_model(vgg_16s, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
                                                                    test_images, test_labels_one_hot)
                                                    predict.predict_model(vgg_16s, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)
                                                else:
                                                    # reseting the model
                                                    vgg_16s_tnb_false.set_weights(old_weights_vgg16s_tnb_false)
                                                    train.fit_model(vgg_16s_tnb_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
                                                                    test_images, test_labels_one_hot)
                                                    predict.predict_model(vgg_16s_tnb_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)'''
            #clear_session()
                                    
            '''print("Training vgg8s")
                                                if (IS_ORIGINAL_TRAINABLE):
                                                    # reseting the model
                                                    vgg_8s.set_weights(old_weights_vgg8s)
                                                    train.fit_model(vgg_8s, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
                                                                    test_images, test_labels_one_hot)
                                                    predict.predict_model(vgg_8s, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)
                                                else:
                                                    # reseting the model
                                                    vgg_8s_tbn_false.set_weights(old_weights_vgg8s_tnb_false)
                                                    train.fit_model(vgg_8s_tbn_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
                                                                    test_images, test_labels_one_hot)
                                                    predict.predict_model(vgg_8s_tbn_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)'''

                                    
            '''print("Training segnet")
                                                if (IS_ORIGINAL_TRAINABLE):
                                                    # reseting the model
                                                    segnet.set_weights(old_weights_segnet)
                                                    train.fit_model(segnet, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
                                                                    test_images, test_labels_one_hot)
                                                    predict.predict_model(segnet, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)
                                                else:
                                                    # reseting the model
                                                    segnet_tnb_false.set_weights(old_weights_segnet_tnb_false)
                                                    train.fit_model(segnet_tnb_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE, train_dataset, val_dataset,
                                                                    test_images, test_labels_one_hot)
                                                    predict.predict_model(segnet_tnb_false, NUM_EPOCHS, BATCH_SIZE, IS_ORIGINAL_TRAINABLE)'''