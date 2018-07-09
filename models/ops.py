import tensorflow as tf


def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.contrib.layers.xavier_initializer(uniform=False)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initial bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def Deconv3D(inputs, filter_size, num_filters, layer_name, stride, out_shape=None, activation=tf.nn.relu):
    """
    Create a 3D transposed-convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param stride: convolution filter stride
    :param out_shape: Tensor of output shape
    :param activation: Activation used at the end of the layer
    :return: The output array
    """
    input_shape = inputs.get_shape().as_list()
    with tf.variable_scope(layer_name):
        kernel_shape = [filter_size, filter_size, filter_size, num_filters, input_shape[-1]]
        # if not len(out_shape.get_shape().as_list()):  # if out_shape is not provided
        #     out_shape = [input_shape[0]] + list(map(lambda x: x * 2, input_shape[1:-1])) + [num_filters]
        weights = weight_variable(layer_name, shape=kernel_shape)
        layer = tf.nn.conv3d_transpose(inputs,
                                       filter=weights,
                                       output_shape=out_shape,
                                       strides=[1, stride, stride, stride, 1],
                                       padding="SAME")
        biases = bias_variable(layer_name, [num_filters])
        layer += biases
        layer = activation(layer)
    return layer
