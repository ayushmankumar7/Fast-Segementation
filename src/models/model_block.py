import tensorflow as tf 



def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding="same", relu = True):

    if (conv_type == 'ds'):
        
        x = tf.keras.layers.SeperableConv2D(kernel, kernel_size, padding = padding, strides = strides)(inputs)
    
    else:
        
        x = tf.keras.layers.Conv2D(kernel, kernel_size, padding = padding, strides = strides)(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    if(relu):
        x = tf.keras.activations.relu(x)

    return x 





