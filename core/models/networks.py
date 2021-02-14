
import tensorflow as tf
from tensorflow.keras import layers
from .utils import *
IMG_SHAPE = (105, 80)
def atari_model(action_size, input_shape=(*IMG_SHAPE, 1)):
    
    inp = layers.Input(shape=input_shape)
    tensor = down_sample_layer(16)(inp)
    tensor = down_sample_layer(32)(tensor)
    tensor = layers.Flatten()(tensor)
    tensor = layers.Dense(256, activation='relu')(tensor)
    tensor = layers.Dense(64, activation='relu')(tensor)
    tensor = layers.Dense(16, activation='relu')(tensor)
    out = layers.Dense(action_size, activation='sigmoid')(tensor)

    model = tf.keras.Model(inp, out)
    return model