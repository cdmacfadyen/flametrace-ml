import tensorflow as tf
from tensorflow import keras

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import plotml
from statistics import mean
import pandas as pd
import os
import cv2

from ml_utils import IMG_SHAPE, IMG_SIZE, reshape_image, timesteps


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) != 0:  
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

##
# Takes arrays of various metrics for each frame and generates more readable 
# results. 
##
def evaluate_all_metrics(overall_accuracy, overall_prediction_list, overall_actual_list, overall_time_per_frame):
  avg_time_per_frame = mean(overall_time_per_frame)
  max_time_per_frame = max(overall_time_per_frame)

  percentage_accuracy = overall_accuracy.result().numpy()
  conf = confusion_matrix(overall_actual_list, overall_prediction_list, labels=[0,1])
  np.savetxt("evaluation/overall_lstm_results.out",conf)
  

  df = pd.DataFrame({
    "accuracy" : percentage_accuracy,
    "average_time_per_frame" : avg_time_per_frame,
    "max_time_per_frame" : max_time_per_frame
  }, index=[0])

  df.to_csv("evaluation/lstm-overall-results.csv".format(video_path))

##
# Takes arrays of various metrics for each frame and generates more readable 
# results. 
##
def evaluate_metrics(accuracy, prediction_list, actual_list, time_per_frame, video_path):
  avg_time_per_frame = mean(time_per_frame)
  
  max_time_per_frame = max(time_per_frame)

  percentage_accuracy = accuracy.result().numpy()
  conf = confusion_matrix(actual_list, prediction_list, labels=[0, 1])
  np.savetxt("evaluation/lstm_results_{}.out".format(video_path),conf)

  first_fire_frame_detected = -1
  first_fire_frame_actual = -1

  for index, label in enumerate(actual_list):
    if int(label) == 1:
      first_fire_frame_actual = index
      break
  
  for index, label in enumerate(prediction_list):
    if int(label) == 1 and index >= first_fire_frame_actual:
      first_fire_frame_detected = index
      break
  
  detection_time = ((first_fire_frame_detected - first_fire_frame_actual) / 3) # diff of 1 = 10 frames = 1/3 sec

  df = pd.DataFrame({
    "accuracy" : percentage_accuracy,
    "average_time_per_frame" : avg_time_per_frame,
    "max_time_per_frame" : max_time_per_frame,
    "detection_time" : detection_time,
    "first_fire_frame" : first_fire_frame_actual,
    "first_fire_frame_detected": first_fire_frame_detected
  }, index=[0])

  df.to_csv("evaluation/{}-results.csv".format(video_path))

  lables_df = pd.DataFrame({
    "predicted" : [pred.numpy() for pred in prediction_list],
    "actual" : actual_list
  })

  lables_df.to_csv("evaluation/{}-lables.csv".format(video_path))



# Load the models 
feature_extractor = tf.keras.models.load_model("models/feature_extractor.h5")
flametrace = tf.keras.models.load_model("models/flametrace_lstm.h5")

# Initialise metrics
accuracy = tf.keras.metrics.Accuracy()
overall_accuracy = tf.keras.metrics.Accuracy()
overall_prediction_list = []
overall_actual_list = []
overall_time_per_frame = []



##
# For every video in the test videos directory folder,
# iterate over it frame by frame. For 
# every frame, run it through the feature extractor
# and then feed these extracted features 
# to the LSTM neural network in blocks of 10 frames. 
# Track the predicted and actual results. 
##
test_dir_path = "./test-videos/"
feature_array = np.zeros((10, 1280))
feature_index = 0
for video_path in os.listdir(test_dir_path):
  temp, path_path = os.path.split(video_path)

  path = test_dir_path + video_path

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
  prediction_list = []
  actual_list = []
  time_per_frame = []
  while(cap.isOpened()):
    ret, frame = cap.read()
    if (not ret):
      break
    start = time.time() # the beginning of processing the frame. 
    tensor = reshape_image(frame)
    print(label_position, end = "\r")
    features = feature_extractor(tensor)
    feature_array[number_of_frames] = tf.reshape(features, [-1])
    number_of_frames += 1
    img_proc_time = time.time() - start

    # If we have the data for 10 frames
    if (number_of_frames) == timesteps:
      feature_tensor = tf.expand_dims(feature_array, 0)
      labels_for_slice = labels[label_position:label_position+timesteps]
      label = np.array([float(labels_for_slice[-1])])

      logits = flametrace(feature_tensor, training=False)    #make prediction
      prediction = tf.argmax(logits, axis=1, output_type=tf.int32)  # find which prediction was made

      accuracy(prediction, label)
      overall_accuracy(prediction, label)

      prediction_list.append(prediction)
      overall_prediction_list.append(prediction)

      actual_list.append(label)
      overall_actual_list.append(label)

      label_position += timesteps

      number_of_frames = 0

    
    end = time.time()
    time_per_frame.append(end - start)  # track how long each frame takes
    overall_time_per_frame.append(end - start)

  cap.release()
  print("\t", end="\r")
  cv2.destroyAllWindows()


  evaluate_metrics(accuracy, prediction_list, actual_list, time_per_frame, video_path)
  accuracy.reset_states() # clear current accuracy data

evaluate_all_metrics(overall_accuracy, overall_prediction_list, overall_actual_list, overall_time_per_frame)