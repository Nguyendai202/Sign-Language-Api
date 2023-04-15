import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops 
from LoadData import trainning_data, testing_data, Y_train, Y_test
from keras.callbacks import TensorBoard 
from datetime import datetime

def convolutional_model(input_shape):
    input_img = tf.keras.Input(shape=input_shape)
    Z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), padding='SAME')(input_img)
    A1 = tf.keras.layers.Activation('relu')(Z1)
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='SAME')(A1)
    Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='SAME')(P1)
    A2 = tf.keras.layers.Activation('relu')(Z2)
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='SAME')(A2)
    F = tf.keras.layers.Flatten()(P2)
    outputs = tf.keras.layers.Dense(units=25, activation='softmax')(F)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

conv_model = convolutional_model((28, 28, 1))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()

train = trainning_data[:-500]
test =trainning_data[-500:]
X = np.array([i[0] for i in train]).reshape([-1, 28, 28, 1])
y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape([-1, 28, 28, 1])
test_y = np.array([i[1] for i in test])
train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(64)

if __name__ == "__main__":
    print("start training model: ")
    log_dir="logs/fit/"+ datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
    history = conv_model.fit(train_dataset, epochs=120, validation_data=test_dataset, callbacks=[tensorboard_callback])
    conv_model.save("my_model_sign_language.h5") 

