import tensorflow as tf 



def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding="same", relu = True):

    if (conv_type == 'ds'):
        
        x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding = padding, strides = strides)(inputs)
    
    else:
        
        x = tf.keras.layers.Conv2D(kernel, kernel_size, padding = padding, strides = strides)(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    if(relu):
        x = tf.keras.activations.relu(x)

    return x 



def _res_bottleneck(inputs, filters, kernel, t, s, r = False):

    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, "conv", tchannel, (1,1), strides =(1,1) )

    x = tf.keras.layers.DepthWiseConv2D(kernel, strides = (s,s), depth_multiplier = 1, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = conv_block(x, 'conv', filters, (1,1), strides =(1,1), padding = 'same', relu = False)

    if r:
        x = tf.keras.keras.layers.add([x, inputs])

    return x





