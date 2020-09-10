import tensorflow as tf 
from model_block import * 


input_layer = tf.keras.layers.Input(shape = (2048, 1024, 3), name = "input_layer")


lds_layer = conv_block(input_layer, 'conv', 32,(3,3), strides = (2,2) )
lds_layer = conv_block(lds_layer, 'ds', 48,(3,3), strides = (2,2) )
lds_layer = conv_block(input_layer,'ds', 64,(3,3), strides = (2,2) )



