import tensorflow as tf 
from model_block import * 


input_layer = tf.keras.layers.Input(shape = (2048, 1024, 3), name = "input_layer")

#Learning to Down Sample 

lds_layer = conv_block(input_layer, 'conv', 32,(3,3), strides = (2,2) )
lds_layer = conv_block(lds_layer, 'ds', 48,(3,3), strides = (2,2) )
lds_layer = conv_block(input_layer,'ds', 64,(3,3), strides = (2,2) )

#Global Feature Extractor

gfe_layer = bottleneck_block(lds_layer, 64, (3,3), t = 6, strides = 2, n =3)
gfe_layer = bottleneck_block(gfe_layer, 96, (3,3), t = 6, strides = 2, n=3)
gfe_layer = bottleneck_block(gfe_layer, 128,(3,3), t = 6, strides = 1, n=3)



