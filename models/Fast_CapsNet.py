from base_model import BaseModel
from layers.Conv_Caps import ConvCapsuleLayer
from layers.FC_Caps import FCCapsuleLayer
from keras import layers
import tensorflow as tf
from ops import *
from layers.ops import squash


class Fast_CapsNet_3D(BaseModel):
    def __init__(self, sess, conf):
        super(Fast_CapsNet_3D, self).__init__(sess, conf)
        self.input_shape = [None, self.conf.height, self.conf.width, self.conf.depth, self.conf.channel]
        self.output_shape = [None, self.conf.num_cls]
        self.create_placeholders()
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            # Layer 1: A 3D conv layer
            conv1 = layers.Conv3D(filters=256, kernel_size=9, strides=1,
                                  padding='valid', activation='relu', name='conv1')(x)

            # Layer 2: Primary Capsule Layer; simply a 3D conv + reshaping
            primary_caps = layers.Conv3D(filters=256, kernel_size=9, strides=2,
                                         padding='valid', activation='relu', name='primary_caps')(conv1)
            _, H, W, D, dim = primary_caps.get_shape()
            primary_caps_reshaped = layers.Reshape((H.value * W.value * D.value, dim.value))(primary_caps)
            caps1_output = squash(primary_caps_reshaped)
            # [?, 512, 256]
            # Layer 3: Digit Capsule Layer; Here is where the routing takes place
            digitcaps_layer = FCCapsuleLayer(num_caps=self.conf.num_cls, caps_dim=self.conf.digit_caps_dim,
                                             routings=3, name='digit_caps')
            self.digit_caps = digitcaps_layer(caps1_output)
            # [?, 10, 16]
            u_hat = digitcaps_layer.get_predictions(caps1_output)

            self.mask()
            self.decoder()

    def decoder(self):
        with tf.variable_scope('Decoder'):
            res1 = Deconv3D(self.cube,
                            filter_size=4,
                            num_filters=16,
                            stride=2,
                            layer_name="deconv_1",
                            out_shape=[self.conf.batch_size, 16, 16, 16, 16])
            self.decoder_output = Deconv3D(res1,
                                           filter_size=4,
                                           num_filters=1,
                                           stride=2,
                                           layer_name="deconv_2",
                                           out_shape=[self.conf.batch_size, 32, 32, 32, 1])
