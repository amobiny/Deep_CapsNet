from base_model import BaseModel
import tensorflow as tf
from models.ops import conv_2d, capsules_init, capsule_conv, capsule_fc


class MatrixCapsNet(BaseModel):
    def __init__(self, sess, conf):
        super(MatrixCapsNet, self).__init__(sess, conf)
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            net = conv_2d(x, 5, 2, 32, 'CONV1')
            # [?, 14, 14, 32]
            pose, act = capsules_init(net, 1, 1, OUT=self.conf.B, padding='VALID',
                                      pose_shape=[4, 4], name='capsule_init')
            # [?, 14, 14, B, 4, 4], [?, 14, 14, B]
            pose, act = capsule_conv(pose, act, K=3, OUT=self.conf.C, stride=2,
                                     iters=self.conf.iter, name='capsule_conv1')
            # [?, 6, 6, C, 4, 4], [?, 6, 6, C]
            pose, act = capsule_conv(pose, act, K=3, OUT=self.conf.D, stride=1,
                                     iters=self.conf.iter, name='capsule_conv2')
            # [?, 4, 4, D, 4, 4], [?, 4, 4, D]
            pose, act = capsule_fc(pose, act, OUT=self.conf.num_cls, iters=self.conf.iter, name='capsule_fc')
            print()

