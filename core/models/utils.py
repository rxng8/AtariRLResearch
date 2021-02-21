
import tensorflow as tf
from tensorflow.keras import layers

def conv_layer(out_channels, strides=1, activation='relu', padding='same'):
    layer = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(4, 4),
        activation=activation,
        padding=padding
    )
    return layer

def dropout_layer(rate=0.5):
    return layers.Dropout(rate)

def max_pooling_layer():
    return layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )

def down_sample_layer(out_channels):
    return tf.keras.Sequential([
        conv_layer(out_channels),
        max_pooling_layer(),
        dropout_layer()
    ])

def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def huber(y_true, y_pred):
    return tf.keras.losses.Huber()(y_true, y_pred)