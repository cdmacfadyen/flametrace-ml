import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os

import pathlib
from ml_utils import CLASS_NAMES, BATCH_SIZE, timesteps

##
# Reads the training features into memory in batches 
# for speedy training. 
## 

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size = shuffle_buffer_size)

  ds = ds.repeat()
  ds = ds.batch(BATCH_SIZE)


  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return ds


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  tensor = tf.io.read_file(file_path)
  tensor = tf.io.parse_tensor(tensor,tf.float32)    
  tensor = tf.math.l2_normalize(tensor) # shape here is 1, 5, 1280
  tensor = tf.reshape(tensor, [timesteps, 1280])
  return tensor, label

def get_label(file_path): # the == maps over the list of class names, making a vector. 
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES   # making one-hot

data_dir = pathlib.Path("./features/")

list_dataset = tf.data.Dataset.list_files(str(data_dir/"*/*")) # what is this doing?



labeled_data = list_dataset.map(process_path, num_parallel_calls = tf.data.experimental.AUTOTUNE)



train_ds = prepare_for_training(labeled_data)


tensor_count = len(list(data_dir.glob('*/*')))