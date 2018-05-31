from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Input
from keras.models import Model

def resnet50(w=240, h=320, c=3):
    input_img = Input((w, h, c))
    return ResNet50(include_top=False, input_tensor=input_img)

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
