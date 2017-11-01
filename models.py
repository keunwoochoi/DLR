from ops import dilated_conv1d, avg_pool2d, get_shape
import tensorflow as tf

def get_local_map(y, ndilation, nchannel, filter_len, dilate_init=1, dilate_scale=2, dtype=0):
    '''
    Args : 
        y - 2D tensor [batch_size, time_len]
        ndilation - int 
        nchannel - int 
        filter_len - int
        dilate_init - int defaults to be 1 
        dilate_scale - int defaults to be 2
        dtype - int defaults to be 0
    Return :
        dtype - 0
            local_map - 4D tensor [batch_size, time_len, ndilationxnchannel, 1]
        dtype - 1
            local_map - 4D tensor [batch_size, time_len, nchannel, ndilation]
        dtype - 2
            local_map - 4D tensor [batch_size*ndilation, time_len, nchannel, 1]
    '''
    batch_size, _ = get_shape(y)
    y_r = tf.expand_dims(y, axis=-1)
    local_map = [] 

    for i in range(ndilation):
        local_map.append(dilated_conv1d(y_r, filter_shape=[filter_len, 1, nchannel],\
                                        stride=1, activation=tf.nn.relu, padding=False,\
                                        dilation_rate=dilate_init*int(dilate_scale**i), scope = "dilated_conv%d"%i))

    min_size = min([item.get_shape().as_list()[1] for item in local_map])
    for i in range(ndilation):
        local_map[i] = local_map[i][:,:min_size,:]
    if dtype ==0: 
        local_map = tf.reshape(tf.concat(local_map, axis=2), [batch_size, min_size, ndilation*nchannel,1])
    elif dtype == 1:
        local_map = tf.reshape(tf.concat(local_map, axis=2), [batch_size, min_size, ndilation, nchannel])
        local_map = tf.transpose(local_map, [0,1,3,2]) # [batch_size, min_size, nchannel, ndilation]
    elif dtype == 2:
        local_map = tf.reshape(tf.concat(local_map, axis=2), [batch_size, min_size, ndilation, nchannel])
        local_map = tf.transpose(local_map, [0,2,1,3]) # [batch_size, ndilation, min_size, nchannel]
        local_map = tf.reshape(local_map, [batch_size*ndilation, min_size, nchannel, 1]) # [batch_size*ndilation, min_size, nchannel, 1]
    return local_map

def dlr13(y):
    '''4 dilation diltation scale 7'''
    local_map = get_local_map(y, ndilation=4, nchannel=32, filter_len=16, dilate_init=1, dilate_scale=13)
    local_map = avg_pool2d(local_map, [256, 1]) 
    return local_map
