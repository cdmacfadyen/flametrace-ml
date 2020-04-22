import tensorflow as tf
from tensorflow import keras

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import plotml

import os
import cv2
from lstm_data_pipeline import train_ds, CLASS_NAMES, BATCH_SIZE, tensor_count

from ml_utils import timesteps

flametrace = keras.models.Sequential()
flametrace.add(keras.layers.LSTM(256, input_shape=(timesteps , 1280)))
flametrace.add(keras.layers.Dense(units=128, activation="relu"))
flametrace.add(keras.layers.Dense(units=64, activation="relu"))
flametrace.add(keras.layers.Dense(units=2, activation="softmax"))
flametrace.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer="adam", metrics=['accuracy' ])
print(flametrace.summary())


num_of_batches = tensor_count // BATCH_SIZE

take = int(0.2 * num_of_batches)

train_set = train_ds.skip(take)
test_set = train_ds.take(take)

steps = (num_of_batches - take)
validation_steps = take


history = flametrace.fit(train_set,
                    epochs=40,
                    validation_data=test_set,
                    validation_steps=validation_steps,
                    steps_per_epoch=steps)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,3.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.savefig("figures/training-acc-graph.png")
flametrace.save("models/flametrace_lstm.h5")