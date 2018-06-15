from keras import initializers, layers
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.utils.conv_utils import conv_output_length

from layers.ops import update_routing


class FCCapsuleLayer(layers.Layer):
    def __init__(self, num_caps, caps_dim, routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(FCCapsuleLayer, self).__init__(**kwargs)
        self.num_caps = num_caps
        self.caps_dim = caps_dim
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.num_in_caps = input_shape[1]
        self.num_out_caps = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_atoms, self.num_caps * self.caps_dim],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[1, 1, self.num_caps, self.caps_dim],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):
        input_transposed = tf.transpose(input_tensor, [3, 0, 1, 2, 4])
        input_shape = K.shape(input_transposed)
        input_tensor_reshaped = K.reshape(input_transposed, [
            input_shape[0] * input_shape[1], self.input_height, self.input_width, self.input_num_atoms])
        input_tensor_reshaped.set_shape((None, self.input_height, self.input_width, self.input_num_atoms))

        conv = K.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
                        padding=self.padding, data_format='channels_last')

        votes_shape = K.shape(conv)
        _, conv_height, conv_width, _ = conv.get_shape()

        votes = K.reshape(conv, [input_shape[1], input_shape[0], votes_shape[1], votes_shape[2],
                                 self.num_caps, self.caps_dim])
        votes.set_shape((None, self.input_num_capsule, conv_height.value, conv_width.value,
                         self.num_caps, self.caps_dim))

        logit_shape = K.stack([
            input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_caps])
        biases_replicated = K.tile(self.b, [conv_height.value, conv_width.value, 1, 1])

        activations = update_routing(votes=votes,
                                     biases=biases_replicated,
                                     logit_shape=logit_shape,
                                     num_dims=6,
                                     input_dim=self.input_num_capsule,
                                     output_dim=self.num_caps,
                                     num_routing=self.routings)

        return activations
