import tensorflow as tf
from base_model import BaseModel
from ops import *
from utils import *


class CapsNet(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=4):
        super(CapsNet, self).__init__(sess, conf)
        self.num_levels = num_levels
        self.k_size = self.conf.filter_size
        self.down_conv_factor = 2
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu',
                                  name='conv1')(x)

            # Reshape layer to be 1 capsule x [filters] atoms
            _, H, W, C = conv1.get_shape()
            conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

            # Layer 1: Primary Capsule: Conv cap with routing 1
            primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                            routings=1, name='primarycaps')(conv1_reshaped)

            # Layer 2: Convolutional Capsule
            secondary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                              routings=3, name='secondarycaps')(primary_caps)

            # Layer 3: Convolutional Capsule


            print()
