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


def bottleneck_block(inputs, filters, kernel, t, strides, n):

    x = _res_bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True)

    return x 


def pyramid_pooling_block(input_tensor, bin_sizes):

    concat_list = [input_tensor]
    w = 64 
    h = 32

    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size = (w//bin_size, h//bin_size), strides = (w//bin_size, h//bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding = 'same')(x)
        x = tf.keras.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)

        concat_list.append(x) 

    return tf.keras.layers.concatenate(concat_list)







