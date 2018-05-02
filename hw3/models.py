import os
from keras.layers import *
from keras.models import Model
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l2

from utils.BilinearUpSampling import *


def vgg16(input_shape, weight_decay, weights_path):
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model_only_convs = Model(img_input, x)
    model_only_convs.load_weights(weights_path, by_name=True)
    for layer in model_only_convs.layers:
        layer.trainable = False

    return img_input, x, model_only_convs


def FCN_Vgg16_32s(input_shape, weight_decay=0., classes=7, weights_path="../vgg16_weights_tf_dim_ordering_tf_kernels.h5", droprate=0.5):
    img_input, vgg_out, _ = vgg16(input_shape, weight_decay, weights_path)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(vgg_out)
    x = Dropout(droprate)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(droprate)(x)

    # Classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1))(x)

    # Upsampling to 512 * 512
    x = Conv2DTranspose(classes, (64, 64), strides=(32, 32), padding='same')(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    model.compile(Adam(1e-4), 'categorical_crossentropy')
    model.summary()
    return model


def FCN_Vgg16_16s(input_shape, weight_decay=0., classes=7, weights_path="../vgg16_weights_tf_dim_ordering_tf_kernels.h5", droprate=0.5):
    img_input, vgg_out, vgg = vgg16(input_shape, weight_decay, weights_path)

    skip_con = Convolution2D(classes, kernel_size=(1,1), padding="same", name="score_pool4")

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(vgg_out)
    x = Dropout(droprate)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(droprate)(x)

    # Classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1))(x)

    # Upsample last output to 32 * 32 and add skip-con from second last pooling
    x = Conv2DTranspose(classes ,kernel_size=(4, 4), strides = (2, 2), padding = "same", name = "score2")(x)
    x = add(inputs = [skip_con(vgg.layers[14].output), x])

    # Upsample sum back to 512 * 512
    x = Conv2DTranspose(classes, (32, 32), strides=(16, 16), padding='same')(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    model.compile(Adam(1e-4), 'categorical_crossentropy')
    model.summary()
    return model


def FCN_Vgg16_8s(input_shape, weight_decay=0., classes=7, weights_path="../vgg16_weights_tf_dim_ordering_tf_kernels.h5", droprate=0.5):
    img_input, vgg_out, vgg = vgg16(input_shape, weight_decay, weights_path)

    skip_con = Convolution2D(classes, kernel_size=(1,1), padding="same", name="score_pool4")

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(vgg_out)
    x = Dropout(droprate)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    x = Dropout(droprate)(x)

    # Classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1))(x)

    # Upsample last output to 32 * 32 and add skip-con from second last pooling
    x = Conv2DTranspose(classes ,kernel_size=(8, 8), strides = (4, 4), padding = "same", name = "score2")(x)
    x = add(inputs = [skip_con(vgg.layers[10].output), x, skip_con(skip_con(vgg.layers[14].output))])

    # Upsample sum back to 512 * 512
    x = Conv2DTranspose(classes, (32, 32), strides=(16, 16), padding='same')(x)
    x = Activation('softmax')(x)
