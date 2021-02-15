import gym
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras import layers

IMG_SHAPE = (105, 80)
BATCH_SIZE = 32

def show_img(data):
    plt.imshow(data)
    plt.axis("off")
    plt.show()

def resize(img: np.ndarray, shape=IMG_SHAPE) -> tf.Tensor:
    return tf.image.resize(
        img, shape, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
        antialias=False, name=None
    )

def down_sample(img: np.ndarray) -> np.ndarray:
    return img[::2, ::2]

def RGB2Gray(img: np.ndarray) -> tf.Tensor:
    return tf.reduce_mean(img, axis=2)

def rescale(img: np.ndarray, range='sigmoid') -> np.ndarray:
    if range == 'sigmoid':
        return img / 255.0
    elif range == 'tanh':
        return ((img / 255.0) - 0.5) * 2
    else:
        print("Please use 'sigmoid' or 'tanh' range type!")

def deprocess(img: tf.Tensor, resize=None) -> tf.Tensor:
    ans = img.copy()
    
    if resize:
        ans = tf.resize(ans, shape=resize)
    
    mn = tf.reduce_min(img)
    mx = tf.reduce_max(img)
    
    # tanh range
    if mn < 0:
        ans = (ans / 2 + 0.5 ) * 255.0
        ans = tf.cast(ans, dtype=tf.uint8)
    # sigmoid range
    elif mx <= 1:
        ans = ans * 255.0
        ans = tf.cast(ans, dtype=tf.uint8)

    return ans

def preprocess(img: np.ndarray) -> tf.Tensor:
    ans = RGB2Gray(img)
    if len(ans.shape) == 2:
        ans = tf.expand_dims(ans, axis=-1)
    ans = resize(ans)
    ans = rescale(ans)
    return ans