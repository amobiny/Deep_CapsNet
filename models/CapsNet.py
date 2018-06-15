from base_model import BaseModel
from layers.Conv_Caps import ConvCapsuleLayer
from layers.FC_Caps import FCCapsuleLayer
from keras import layers
import tensorflow as tf


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
            # Layer 1: A 2D conv layer
            conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1,
                                  padding='same', activation='relu', name='conv1')(x)

            # Reshape layer to be 1 capsule x caps_dim(=filters)
            _, H, W, C = conv1.get_shape()
            conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

            # Layer 1: Primary Capsule: Conv cap with routing 1
            primary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=2, caps_dim=16, strides=2, padding='same',
                                            routings=1, name='primarycaps')(conv1_reshaped)

            # Layer 2: Convolutional Capsule
            secondary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=4, caps_dim=16, strides=1, padding='same',
                                              routings=3, name='secondarycaps')(primary_caps)
            _, H, W, D, dim = secondary_caps.get_shape()
            sec_cap_reshaped = layers.Reshape((H.value*W.value*D.value, C.value))(conv1)

            # Layer 3: Fully-connected Capsule
            digit_caps_dim = 16
            digit_caps = FCCapsuleLayer(num_caps=self.conf.num_cls, caps_dim=digit_caps_dim,
                                        routings=3, name='secondarycaps')(sec_cap_reshaped)
            # [batch_size, 10, 16]

            # Decoder
            with tf.variable_scope('Masking'):
                epsilon = 1e-9
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=2, keep_dims=True) + epsilon)
                # [?, 10, 1]

                y_prob_argmax = tf.to_int32(tf.argmax(self.v_length, axis=1))
                # [?, 1]
                self.y_pred = tf.reshape(y_prob_argmax, shape=(-1))
                # [?] (predicted labels)
                y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.num_cls)
                # [?, 10] (one-hot-encoded predicted labels)

                reconst_targets = tf.cond(self.mask_with_labels,  # condition
                                          lambda: self.y,  # if True (Training)
                                          lambda: y_pred_ohe,  # if False (Test)
                                          name="reconstruction_targets")
                # [?, 10]

                caps2_output_masked = tf.multiply(tf.squeeze(digit_caps), tf.expand_dims(reconst_targets, -1))
                # [?, 10, 16]

                decoder_input = tf.reshape(caps2_output_masked, [-1, self.conf.num_cls*digit_caps_dim])
                # [?, 160]

            with tf.variable_scope('Decoder'):
                fc1 = tf.layers.dense(decoder_input, n_hidden1, activation=tf.nn.relu, name="FC1")
                # [batch_size, 512]
                fc2 = tf.layers.dense(fc1, n_hidden2, activation=tf.nn.relu, name="FC2")
                # [batch_size, 1024]
                self.decoder_output = tf.layers.dense(fc2, n_output, activation=tf.nn.sigmoid, name="FC3")
                # [batch_size, 784]
            print()
