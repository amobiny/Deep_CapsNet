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
            self.digit_caps = digitcaps_layer(caps1_output)  # [?, 10, 16]
            u_hat = digitcaps_layer.get_predictions(caps1_output)  # [?, 2, 512, 16]
            u_hat_shape = u_hat.get_shape().as_list()
            self.img_s = int(round(u_hat_shape[2] ** (1. / 3)))
            self.u_hat = layers.Reshape(
                (self.conf.num_cls, self.img_s, self.img_s, self.img_s, 1, self.conf.digit_caps_dim))(u_hat)
            # self.u_hat = tf.transpose(u_hat, perm=[1, 0, 2, 3, 4, 5, 6])
            # u_hat: [2, ?, 8, 8, 8, 1, 16]
            self.decoder()

    def decoder(self):
        with tf.variable_scope('Decoder'):
            epsilon = 1e-9
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=2, keep_dims=True) + epsilon)
            # [batch_size, 2, 1]
            indices = tf.argmax(tf.squeeze(self.v_length), axis=1)
            self.y_prob = tf.nn.softmax(tf.squeeze(self.v_length))
            self.y_pred = tf.squeeze(tf.to_int32(tf.argmax(self.v_length, axis=1)))
            # [batch_size] (predicted labels)
            y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.num_cls)
            # [batch_size, 2] (one-hot-encoded predicted labels)

            reconst_targets = tf.cond(self.mask_with_labels,  # condition
                                      lambda: self.y,  # if True (Training)
                                      lambda: y_pred_ohe,  # if False (Test)
                                      name="reconstruction_targets")
            # [batch_size, 2]
            reconst_targets = tf.reshape(reconst_targets, (-1, 1, 1, 1, self.conf.num_cls))
            # [batch_size, 1, 1, 1, 2]
            reconst_targets = tf.tile(reconst_targets, (1, self.img_s, self.img_s, self.img_s, 1))
            # [batch_size, 8, 8, 8, 2]

            num_partitions = self.conf.batch_size
            partitions = tf.range(num_partitions)
            u_list = tf.dynamic_partition(self.u_hat, partitions, num_partitions, name='uhat_dynamic_unstack')
            ind_list = tf.dynamic_partition(indices, partitions, num_partitions, name='ind_dynamic_unstack')

            a = tf.stack([tf.gather_nd(tf.squeeze(mat, axis=0), [[ind]]) for mat, ind in zip(u_list, ind_list)])
            # [?, 1, 8, 8, 8, 1, 16]
            feat = tf.reshape(a, (-1, self.img_s, self.img_s, self.img_s, self.conf.digit_caps_dim))
            # [?, 8, 8, 8, 16]
            self.cube = tf.concat([feat, reconst_targets], axis=-1)
            # [?, 8, 8, 8, 18]

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

    def configure_summary(self):
        recon_img = tf.reshape(self.decoder_output[:, :, :, 16, :],
                               shape=(-1, self.conf.height, self.conf.width, self.conf.channel))
        orig_image = tf.reshape(self.x[:, :, :, 16, :],
                                shape=(-1, self.conf.height, self.conf.width, self.conf.channel))
        summary_list = [tf.summary.scalar('Loss/total_loss', self.mean_loss),
                        tf.summary.scalar('Accuracy/average_accuracy', self.mean_accuracy),
                        tf.summary.image('original', orig_image),
                        tf.summary.image('reconstructed', recon_img)]
        self.merged_summary = tf.summary.merge(summary_list)

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        if self.conf.data == 'mnist':
            from MNISTLoader import DataLoader
        elif self.conf.data == 'nodule':
            from DataLoader import DataLoader

        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_validation()
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('*' * 50)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
            print('*' * 50)
        else:
            print('*' * 50)
            print('----> Start Training')
            print('*' * 50)
        self.num_train_batch = int(self.data_reader.y_train.shape[0] / self.conf.batch_size)
        self.num_val_batch = int(self.data_reader.y_valid.shape[0] / self.conf.val_batch_size)
        for epoch in range(self.conf.max_epoch):
            self.data_reader.randomize()
            for train_step in range(self.num_train_batch):
                start = train_step * self.conf.batch_size
                end = (train_step + 1) * self.conf.batch_size
                global_step = epoch * self.num_train_batch + train_step
                x_batch, y_batch = self.data_reader.next_batch(start, end)
                feed_dict = {self.x: x_batch, self.y: y_batch, self.mask_with_labels: True}
                if train_step % self.conf.SUMMARY_FREQ == 0:
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    self.save_summary(summary, global_step, mode='train')
                    print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                else:
                    self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

            self.evaluate(epoch)
            self.save(epoch)

    def evaluate(self, epoch, global_step):
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_val_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.x: x_val, self.y: y_val, self.mask_with_labels: False}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, global_step, mode='valid')
        print('-' * 30 + 'Validation' + '-' * 30)
        print('Epoch #{0} : val_loss= {1:.4f}, val_acc={2:.01%}'.format(epoch, valid_loss, valid_acc))
        print('-' * 70)