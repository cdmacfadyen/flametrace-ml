import tensorflow as tf
import numpy as np

CLASS_NAMES = np.array(["no-fire", "fire"])
BATCH_SIZE = 64
timesteps = 10

IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def reshape_image(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 127.5) - 1

  image = tf.convert_to_tensor(image, dtype=tf.float32)
  image = tf.expand_dims(image, 0)    # this adds an extra dimension for "batch"
  return image