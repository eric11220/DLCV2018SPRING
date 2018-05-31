import keras.backend as K
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Input, Lambda , LSTM
from keras.models import Model

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
    print(input_dim)
    input_vec = Input(input_dim)
    out = LSTM(lstm_dim, dropout=droprate, recurrent_dropout=droprate, return_sequences=return_sequences)(input_vec)
    out = Dense(n_class, activation='softmax')(out)

    model = Model(inputs=input_vec, outputs=out)
    model.compile(loss="categorical_crossentropy",
                  metrics=['accuracy'],
                  optimizer='adam')
    model.summary()
    return model
