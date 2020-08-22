from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Softmax, Activation, Reshape, BatchNormalization, Input

from .custom.custom_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
import config


def Segnet(pretrained=True, tnb_extractor=True):
    """#Building the model

    > Here we first download the VGG16 feature extractor model (and for this purpose we remove the classification block, i.e. the dense layers)

    > After, we build the Segnet encoder model by adding the particular segnet layers to the original VGG16 feature extractor, so that we can take advantage of the pretrained weights from the imagenet dataset.

    > Finally, we build the Segnet decoder and merge it to the encoder to form the complete Segnet model.
    """

    if (pretrained):
      feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS))
      feature_extractor.trainable = tnb_extractor
    else:
      feature_extractor = VGG16(include_top=False, input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS))
      feature_extractor.trainable = True

    pool_size = (2, 2)

    """# Encoder"""

    model_input = Input(shape=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.CHANNELS))
    x = (feature_extractor.get_layer('block1_conv1'))(model_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = feature_extractor.get_layer('block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = (feature_extractor.get_layer('block2_conv1'))(pool_1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = feature_extractor.get_layer('block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = (feature_extractor.get_layer('block3_conv1'))(pool_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = feature_extractor.get_layer('block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = feature_extractor.get_layer('block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = (feature_extractor.get_layer('block4_conv1'))(pool_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = feature_extractor.get_layer('block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = feature_extractor.get_layer('block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(x)

    x = (feature_extractor.get_layer('block5_conv1'))(pool_4)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = feature_extractor.get_layer('block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = feature_extractor.get_layer('block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(x)

    """### Decoder"""

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    x = (Conv2D(512, (3, 3), padding="same"))(unpool_1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = (Conv2D(512, (3, 3), padding="same"))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = (Conv2D(512, (3, 3), padding="same"))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    unpool_2 = MaxUnpooling2D(pool_size)([x, mask_4])

    x = (Conv2D(512, (3, 3), padding="same"))(unpool_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = (Conv2D(512, (3, 3), padding="same"))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = (Conv2D(256, (3, 3), padding="same"))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    unpool_3 = MaxUnpooling2D(pool_size)([x, mask_3])

    x = (Conv2D(256, (3, 3), padding="same"))(unpool_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = (Conv2D(256, (3, 3), padding="same"))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = (Conv2D(128, (3, 3), padding="same"))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    unpool_4 = MaxUnpooling2D(pool_size)([x, mask_2])

    x = (Conv2D(128, (3, 3), padding="same"))(unpool_4)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = (Conv2D(64, (3, 3), padding="same"))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    unpool_5 = MaxUnpooling2D(pool_size)([x, mask_1])

    x = (Conv2D(64, (3, 3), padding="same"))(unpool_5)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = (Conv2D(5, (1, 1), padding="valid"))(x)
    x = BatchNormalization()(x)
    #x = Reshape((config.IMAGE_WIDTH * config.IMAGE_HEIGHT, config.NUM_CLASSES), input_shape=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.NUM_CLASSES))(x)

    model_output = Activation('softmax')(x)

    model = Model(inputs=model_input, outputs=model_output, name="SegNet")

    model.summary()

    return model