import tensorflow as tf 
from model_block import * 


input_layer = tf.keras.layers.Input(shape = (2048, 1024, 3), name = "input_layer")

#Learning to Down Sample 

lds_layer = conv_block(input_layer, 'conv', 32,(3,3), strides = (2,2) )
lds_layer = conv_block(lds_layer, 'ds', 48,(3,3), strides = (2,2) )
lds_layer = conv_block(lds_layer,'ds', 64,(3,3), strides = (2,2) )

#Global Feature Extractor

gfe_layer = bottleneck_block(lds_layer, 64, (3,3), t = 6, strides = 2, n =3)
gfe_layer = bottleneck_block(gfe_layer, 96, (3,3), t = 6, strides = 2, n=3)
gfe_layer = bottleneck_block(gfe_layer, 128,(3,3), t = 6, strides = 1, n=3)


# PPM 

gfe_layer = pyramid_pooling_block(gfe_layer, [2,4,6,8])

# Feature Fusion 

ff_layer1 = conv_block(lds_layer, 'conv', 128, (1,1), padding='same', strides = (1,1), relu=False )

ff_layer2 = tf.keras.layers.UpSampling2D((4,4))(gfe_layer)
ff_layer2 = tf.keras.layers.DepthwiseConv2D(128, strides = (1,1), depth_multiplier=1, padding='same')(ff_layer2)
ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
ff_layer2 = tf.keras.activations.relu(ff_layer2)
ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation = None)(ff_layer2)


ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
ff_final = tf.keras.layers.BatchNormalization()(ff_final)
ff_final = tf.keras.activations.relu(ff_final)
