from base_model import BaseModel
import tensorflow as tf
from models.utils.ops import conv_2d, capsules_init, capsule_conv, capsule_fc


class MatrixCapsNet(BaseModel):
    def __init__(self, sess, conf):
        super(MatrixCapsNet, self).__init__(sess, conf)
        self.is_train = True
        self.build_network(self.x)
        self.configure_network()


    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            net, summary = conv_2d(x, 5, 2, self.conf.A, 'CONV1', batch_norm=True, is_train=self.is_train)
            # [?, 14, 14, A]
            self.summary_list.append(summary)

            pose, act, summary_list = capsules_init(net, 1, 1, OUT=self.conf.B, padding='VALID',
                                                    pose_shape=[4, 4], name='capsule_init')
            # [?, 14, 14, B, 4, 4], [?, 14, 14, B]
            for summary in summary_list:
                self.summary_list.append(summary)

            pose, act, summary_list = capsule_conv(pose, act, K=3, OUT=self.conf.C, stride=2,
                                                   iters=self.conf.iter, name='capsule_conv1')
            # [?, 6, 6, C, 4, 4], [?, 6, 6, C]
            for summary in summary_list:
                self.summary_list.append(summary)

            pose, act, summary_list = capsule_conv(pose, act, K=3, OUT=self.conf.D, stride=1,
                                                   iters=self.conf.iter, name='capsule_conv2')
            # [?, 4, 4, D, 4, 4], [?, 4, 4, D]
            for summary in summary_list:
                self.summary_list.append(summary)

            self.pose, self.act, summary_list = capsule_fc(pose, act, OUT=self.conf.num_cls,
                                                           iters=self.conf.iter, name='capsule_fc')
            # [?, num_cls, 4, 4], [?, num_cls]
            for summary in summary_list:
                self.summary_list.append(summary)

            self.y_pred = tf.to_int32(tf.argmax(self.act, axis=1))
