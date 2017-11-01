import tensorflow as tf
import numpy as np

def get_shape(x):
    '''get the shape of tensor as list'''
    return x.get_shape().as_list()

def avg_pool2d(x, per_avg):
    '''
    avg_pool2d
    Args:
        x - 4d tensor
            'NHWC' batch, height, width, channel
        per_avg - list with two ints
    '''
    return tf.nn.pool(x, window_shape = per_avg, pooling_type = 'AVG', strides = per_avg, padding='VALID')

def dilated_conv1d(x, filter_shape, stride = 1, dilation_rate = 1, activation = None, padding=True, scope=None):
    '''dilated convolution 1d
    Args :
        x - 3D tensor
            'NLC' batch, length, inchannel
        filte_shape - list with 3 numbers
            [filter_length, inchannel, outchannel]
        stride - int
            stride in filter_length direction defaults to be 1
        dilation_rate - int
            dilation rate in convolution defaults to be 1
        activation - activation function
            defaults to be None
        padding - bool defaults to be False
            True => padding
            False => not padding
        scope - string
            defaults to be None
    Return :
        dilate_conv - 3D tensor shape
            padding(True)
            'NLC' batch, [(length - (dilation_rate*(filter_length-1)+1))/stride], outchannel
    '''
    assert get_shape(x)[2]==filter_shape[1], "Number of inchannel %d and %d should be same"%(get_shape(x)[2], filter_shape[1])

    with tf.variable_scope(scope or 'dilated_conv'):
        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        fan_in = filter_shape[0]*filter_shape[1]
        fan_out = filter_shape[0]*filter_shape[1]*filter_shape[2]
        stddev = np.sqrt(2.0/(fan_in+fan_out))

        w = tf.get_variable(name='weight', shape=filter_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=stddev))
        b = tf.get_variable(name='bias', shape=filter_shape[-1], initializer=tf.constant_initializer(0.01))

        dilate_conv = tf.nn.convolution(x, w, padding=padding, strides=[stride], dilation_rate=[dilation_rate])
        if activation is None:
            return dilate_conv+b
        else:
            return activation(dilate_conv+b)
