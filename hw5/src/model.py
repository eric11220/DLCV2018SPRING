import keras.backend as K
import matplotlib
matplotlib.use('Agg')

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Input, Lambda , LSTM, Masking
from keras.models import Model
from matplotlib import pyplot as plt

def resnet50(w=240, h=320, c=3):
    input_img = Input((w, h, c))
    mean = Input((w, h, c))

    subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
                            output_shape=lambda shapes: shapes[0])
    norm_input_img = subtract_layer([input_img, mean])

    resnet = ResNet50(include_top=False, input_tensor=norm_input_img)
    resnet.summary()
    return resnet

def action_fc_model(input_dim=2048, n_class=11, droprate=0.2):
    input_vec = Input((input_dim, ))
    out = Dense(512, activation='relu')(input_vec)
    out = Dense(128, activation='relu')(out)
    out = Dropout(droprate)(out)
    out = Dense(128, activation='relu')(out)
    out = Dropout(droprate)(out)
    out = Dense(n_class, activation='softmax')(out)

    model = Model(inputs=input_vec, outputs=out)
    model.compile(loss="categorical_crossentropy",
                  metrics=['accuracy'],
                  optimizer='adam')
    model.summary()
    return model

def rnn(input_dim, lstm_dim=256, n_class=11, droprate=0.2, return_sequences=False):
    input_vec = Input(input_dim)
    out = Masking(mask_value=0., input_shape=input_dim)(input_vec)

    out = LSTM(lstm_dim, dropout=droprate, recurrent_dropout=droprate, return_sequences=return_sequences)(out)
    out = Dense(n_class, activation='softmax')(out)

    model = Model(inputs=input_vec, outputs=out)
    model.compile(loss="categorical_crossentropy",
                  metrics=['accuracy'],
                  optimizer='adam')
    model.summary()
    return model

def rnn_feat_extractor(model):
    inputs = model.input
    outputs = model.layers[2].output
    feature_extractor = K.function([inputs]+ [K.learning_phase()], [outputs])
    return feature_extractor

def plot_training_curve(acc, loss, val_acc, val_loss, path):
    plt.clf()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss')
    a, = ax1.plot(loss, label="Training loss", color='C0')
    b, = ax1.plot(val_loss, label="Validation loss", color='C0')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    c, = ax2.plot(acc, label="Training acc.", color='C1')
    d, = ax2.plot(val_acc, label="Validation acc.", color='C1')

    plt.legend(handles=[a,b,c,d])
    plt.tight_layout()
    plt.savefig(path)
