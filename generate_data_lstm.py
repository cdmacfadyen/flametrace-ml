import tensorflow as tf
from tensorflow import keras

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import os
import cv2

from ml_utils import IMG_SHAPE, IMG_SIZE, timesteps, reshape_image


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) != 0:  # sometimes crashes without this
  tf.config.experimental.set_memory_growth(physical_devices[0], True)


model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights="imagenet")
feature_extractor = tf.keras.Sequential([
  model,
  keras.layers.MaxPooling2D(pool_size=4),
  keras.layers.Flatten()
])

feature_extractor.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.1),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

feature_extractor.save("models/feature_extractor.h5")
# print(feature_extractor.summary())



##
# For every video in the training set, read 10 frames at a time 
# and serialise them into either the fire or non-fire set. 
##
dir_path = "./train-videos/"

input("This will remove all features: ok?")

os.system("rm features/fire/*")
os.system("rm features/no-fire/*")
for path in os.listdir(dir_path):
  temp, path_path = os.path.split(path)

  path = dir_path + path

  labels = ""
  label_name = path_path.split(".")[0] + ".lbl"
  label_path = "../heuristic-cv/results/{}".format(label_name)

  with open(label_path) as f:
    labels=f.readline()

  cap = cv2.VideoCapture(path)
  number_of_frames = 0
  count_for_test = 0
  label_position = 0
  feature_list = []
  print("Started feature extraction for {}".format(path))
  print("Number of frames ", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

  feature_array_list = []
  label_list = []
  while(cap.isOpened()):
    ret, frame = cap.read()
    if (not ret):
      break
    tensor = reshape_image(frame)
    print(label_position, end = "\r")
    features = feature_extractor(tensor)
    feature_list.append(tf.reshape(features, [-1]))
    number_of_frames += 1
    if (number_of_frames) == timesteps:
      feature_array = np.array(feature_list)
      feature_array = tf.expand_dims(feature_array, 0)
      labels_for_slice = labels[label_position:label_position+timesteps]
      label = np.array([float(labels_for_slice[-1])])
      directory = "fire" if labels_for_slice[-1] == "1" else "no-fire"

      count_for_test += 1
      tf.io.write_file("{}/{}/{}-{}".format("features", directory, path_path, label_position), 
                                          tf.io.serialize_tensor(feature_array))


      label_position += timesteps

      number_of_frames = 0

      feature_list.clear()
  cap.release()
  print("\t", end="\r")
  cv2.destroyAllWindows()
