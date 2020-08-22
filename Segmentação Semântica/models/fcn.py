from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121

from tensorflow.keras.layers import Input, Add, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, Softmax, Reshape, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

#Personal modules
import config


def DenseNet(pretrained=True, tnb_extractor=True):

    DenseNet121().summary()
    VGG16().summary()

    if (pretrained):
      feature_extractor = DenseNet121(weights='imagenet', include_top=False, input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS))
      feature_extractor.trainable = tnb_extractor
    else:
      feature_extractor = DenseNet121(include_top=False, input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS))
      feature_extractor.trainable = True

    feature_extractor.summary()

    x = Conv2D(1024, (7, 7), activation='relu', padding='same')(feature_extractor.output)
    x = Dropout(0.5)(x)
    '''x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
                x = Dropout(0.5)(x)'''
    x = Conv2D(5, (1, 1), activation='linear')(x)
    x = Conv2DTranspose(5, kernel_size=(64, 64), strides=(32, 32), padding='same')(x)
    # x = Reshape((IMAGE_WIDTH*IMAGE_HEIGHT, -1))(x)
    outputs = Softmax(axis=-1)(x)
    '''outputs = Lambda(prob_to_labels)(x)
    outputs = Reshape((224, 224))(outputs)'''

    model = Model(inputs=feature_extractor.inputs, outputs=outputs, name="DenseNet")

    model.summary()

    return model

def FCN_VGG16_32s(pretrained=True, tnb_extractor=True):
    if (pretrained):
      feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS))
      feature_extractor.trainable = tnb_extractor
    else:
      feature_extractor = VGG16(include_top=False, input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS))
      feature_extractor.trainable = True

    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(feature_extractor.output)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(5, (1, 1), activation='linear')(x)
    x = Conv2DTranspose(5, kernel_size=(64, 64), strides=(32, 32), padding='same')(x)
    # x = Reshape((IMAGE_WIDTH*IMAGE_HEIGHT, -1))(x)
    outputs = Softmax(axis=-1)(x)
    '''outputs = Lambda(prob_to_labels)(x)
    outputs = Reshape((224, 224))(outputs)'''

    model = Model(inputs=feature_extractor.inputs, outputs=outputs, name="FCN_VGG16_32s")

    model.summary()

    return model

def FCN_VGG16_16s(pretrained=True, tnb_extractor=True):
    wd = 0.1
    kr = regularizers.l2
    in1 = Input(shape=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.CHANNELS))
    # ki = 'he_normal'
    ki = 'glorot_uniform'


    if (pretrained):
      feature_extractor = VGG16(weights='imagenet', include_top=False, input_tensor=in1)
      feature_extractor.trainable = tnb_extractor
    else:
      feature_extractor = VGG16(include_top=False, input_tensor=in1)
      feature_extractor.trainable = True

    pool_5 = feature_extractor.get_layer('block5_pool')

    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(pool_5.output)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    score_32s = Conv2D(5, (1, 1))(x)
    # At this point we have the normal output of the FCN 32s

    # Skip connection 1
    # Upscaling the last pooling output so that it can be further summed with the pool4 layer
    upscore2 = Conv2DTranspose(config.NUM_CLASSES, 4,
                               strides=(2, 2),
                               padding='same',
                               kernel_regularizer=kr(wd),
                               kernel_initializer=ki,
                               use_bias=False,
                               name='upscore2')(score_32s)

    # Getting the pool4 layer
    pool_4 = feature_extractor.get_layer('block4_pool')
    # Applying a 1x1 convolution to generate as many feature maps as there are classes (output of the upsampled conv7 layer)
    score_pool4 = Conv2D(config.NUM_CLASSES, 1,
                         kernel_regularizer=kr(wd),
                         use_bias=True)(pool_4.output)

    # Adding the upsampled conv7 layer to the pool4 layer
    fuse_pool4 = Add()([upscore2, score_pool4])

    upscore8 = Conv2DTranspose(config.NUM_CLASSES, 32,
                               strides=(16, 16),
                               padding='same',
                               kernel_regularizer=kr(wd),
                               kernel_initializer=ki,
                               use_bias=False,
                               name='upscore8')(fuse_pool4)

    #reshape = Reshape((config.IMAGE_HEIGHT * config.IMAGE_WIDTH, -1))(upscore8)
    output = Softmax(axis=-1)(upscore8)

    model = Model(in1, output, name="FCN_VGG16_16s")

    model.summary()

    return model

def FCN_VGG16_8s(pretrained=True, tnb_extractor=True):
    wd = 0.1
    kr = regularizers.l2
    in1 = Input(shape=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.CHANNELS))
    # ki = 'he_normal'
    ki = 'glorot_uniform'

    if (pretrained):
      feature_extractor = VGG16(weights='imagenet', include_top=False, input_tensor=in1)
      feature_extractor.trainable = tnb_extractor
    else:
      feature_extractor = VGG16(include_top=False, input_tensor=in1)
      feature_extractor.trainable = True

    pool_5 = feature_extractor.get_layer('block5_pool')

    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(pool_5.output)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    score_32s = Conv2D(5, (1, 1))(x)
    # At this point we have the normal output of the FCN 32s

    # Skip connection 1
    # Upscaling the last pooling output so that it can be further summed with the pool4 layer
    upscore2 = Conv2DTranspose(config.NUM_CLASSES, 4,
                               strides=(2, 2),
                               padding='same',
                               kernel_regularizer=kr(wd),
                               kernel_initializer=ki,
                               use_bias=False,
                               name='upscore2')(score_32s)

    # Getting the pool4 layer
    pool_4 = feature_extractor.get_layer('block4_pool')
    # Applying a 1x1 convolution to generate as many feature maps as there are classes (output of the upsampled conv7 layer)
    score_pool4 = Conv2D(config.NUM_CLASSES, 1,
                         kernel_regularizer=kr(wd),
                         use_bias=True)(pool_4.output)

    # Adding the upsampled conv7 layer to the pool4 layer
    fuse_pool4 = Add()([upscore2, score_pool4])

    # Skip connection 2
    upscore_pool4 = Conv2DTranspose(config.NUM_CLASSES, 4,
                                    strides=(2, 2),
                                    padding='same',
                                    kernel_regularizer=kr(wd),
                                    kernel_initializer=ki,
                                    use_bias=False,
                                    name='upscore_pool4')(fuse_pool4)

    pool_3 = feature_extractor.get_layer('block3_pool')
    score_pool3 = Conv2D(config.NUM_CLASSES, 1, kernel_regularizer=kr(wd), use_bias=True)(pool_3.output)
    fuse_pool3 = Add()([upscore_pool4, score_pool3])

    upscore8 = Conv2DTranspose(config.NUM_CLASSES, 16,
                               strides=(8, 8),
                               padding='same',
                               kernel_regularizer=kr(wd),
                               kernel_initializer=ki,
                               use_bias=False,
                               name='upscore8')(fuse_pool3)

    #reshape = Reshape((config.IMAGE_HEIGHT * config.IMAGE_WIDTH, -1))(upscore8)
    output = Softmax(axis=-1)(upscore8)
    #output = Lambda(predictions_to_rgb)(output)

    model = Model(in1, output, name="FCN_VGG16_8s")

    model.summary()

    return model