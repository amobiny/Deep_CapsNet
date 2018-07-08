from base_model import BaseModel
from layers.Conv_Caps import ConvCapsuleLayer
from layers.FC_Caps import FCCapsuleLayer
from keras import layers
import tensorflow as tf


class CapsNet(BaseModel):
    def __init__(self, sess, conf):
        super(CapsNet, self).__init__(sess, conf)
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            # Layer 1: A 2D conv layer
            conv1 = layers.Conv2D(filters=64, kernel_size=9, strides=1,
                                  padding='same', activation='relu', name='conv1')(x)

            # Reshape layer to be 1 capsule x caps_dim(=filters)
            _, H, W, C = conv1.get_shape()
            conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

            # Layer 2: Convolutional Capsule
            primary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=4, caps_dim=16, strides=2, padding='same',
                                            routings=1, name='primarycaps')(conv1_reshaped)

            # Layer 3: Convolutional Capsule
            # secondary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=4, caps_dim=8, strides=2, padding='same',
            #                                   routings=3, name='secondarycaps')(primary_caps)
            _, H, W, D, dim = primary_caps.get_shape()
            sec_cap_reshaped = layers.Reshape((H.value * W.value * D.value, dim.value))(primary_caps)

            # Layer 4: Fully-connected Capsule
            self.digit_caps, _ = FCCapsuleLayer(num_caps=self.conf.num_cls, caps_dim=self.conf.digit_caps_dim,
                                                routings=1, name='secondarycaps')(sec_cap_reshaped)
            # [?, 10, 16]

            self.mask()
            self.decoder()

    def decoder(self):
        with tf.variable_scope('Decoder'):
            decoder_input = tf.reshape(self.output_masked, [-1, self.conf.num_cls * self.conf.digit_caps_dim])
            # [?, 160]
            fc1 = tf.layers.dense(decoder_input, self.conf.h1, activation=tf.nn.relu, name="FC1")
            # [?, 512]
            fc2 = tf.layers.dense(fc1, self.conf.h2, activation=tf.nn.relu, name="FC2")
            # [?, 1024]
            self.decoder_output = tf.layers.dense(fc2, self.conf.width * self.conf.height,
                                                  activation=tf.nn.sigmoid, name="FC3")
            # [?, 784]
