import tensorflow as tf 
import model_block


input_layer = tf.keras.layers.Input(shape = (2048, 1024, 3), name = "input_layer")


