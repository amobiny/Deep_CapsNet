import tensorflow as tf
import os
import numpy as np
from MNISTLoader import DataLoader


class BaseModel(object):
    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.input_shape = [None, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [None, self.conf.num_cls]
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.float32, self.output_shape, name='annotation')
            self.mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

    def mask(self):
        with tf.variable_scope('Masking'):
            epsilon = 1e-9
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=2, keepdims=True) + epsilon)
            # [?, 10, 1]

            y_prob_argmax = tf.to_int32(tf.argmax(self.v_length, axis=1))
            # [?, 1]
            self.y_pred = tf.squeeze(y_prob_argmax)
            # [?] (predicted labels)
            y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.num_cls)
            # [?, 10] (one-hot-encoded predicted labels)

            reconst_targets = tf.cond(self.mask_with_labels,  # condition
                                      lambda: self.y,  # if True (Training)
                                      lambda: y_pred_ohe,  # if False (Test)
                                      name="reconstruction_targets")
            # [?, 10]

            self.output_masked = tf.multiply(self.digit_caps, tf.expand_dims(reconst_targets, -1))
            # [?, 10, 16]

    def loss_func(self):
        # 1. The margin loss
        with tf.variable_scope('Margin_Loss'):
            # max(0, m_plus-||v_c||)^2
            present_error = tf.square(tf.maximum(0., self.conf.m_plus - self.v_length))
            # [?, 10, 1]

            # max(0, ||v_c||-m_minus)^2
            absent_error = tf.square(tf.maximum(0., self.v_length - self.conf.m_minus))
            # [?, 10, 1]

            # reshape: [?, 10, 1] => [?, 10]
            present_error = tf.squeeze(present_error)
            absent_error = tf.squeeze(absent_error)

            T_c = self.y
            # [?, 10]
            L_c = T_c * present_error + self.conf.lambda_val * (1 - T_c) * absent_error
            # [?, 10]
            self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1), name="margin_loss")

        # 2. The reconstruction loss
        with tf.variable_scope('Reconstruction_Loss'):
            orgin = tf.reshape(self.x, shape=(-1, self.conf.height * self.conf.width))
            squared = tf.square(self.decoder_output - orgin)
            self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        with tf.variable_scope('Total_Loss'):
            self.total_loss = self.margin_loss + self.conf.alpha * self.reconstruction_err
            self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)


    def accuracy_func(self):
        with tf.variable_scope('Accuracy'):
            correct_prediction = tf.equal(tf.to_int32(tf.argmax(self.y, axis=1)), self.y_pred)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        with tf.name_scope('Optimizer'):
            with tf.name_scope('Learning_rate_decay'):
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                              trainable=False)
                steps_per_epoch = self.conf.num_tr // self.conf.batch_size
                learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                           global_step,
                                                           steps_per_epoch,
                                                           0.97,
                                                           staircase=True)
                self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        recon_img = tf.reshape(self.decoder_output, shape=(-1, self.conf.height, self.conf.width, self.conf.channel))
        summary_list = [tf.summary.scalar('Loss/total_loss', self.mean_loss),
                        tf.summary.scalar('Accuracy/average_accuracy', self.mean_accuracy),
                        tf.summary.image('original', self.x),
                        tf.summary.image('reconstructed', recon_img)]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, mode):
        # print('----> Summarizing at step {}'.format(step))
        if mode == 'train':
            self.train_writer.add_summary(summary, step)
        elif mode == 'valid':
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
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
        self.num_val_batch = int(self.data_reader.y_valid.shape[0] / self.conf.val_batch_size)
        for train_step in range(1, self.conf.max_step + 1):
            if train_step % self.conf.SUMMARY_FREQ == 0:
                x_batch, y_batch = self.data_reader.next_random_batch()
                feed_dict = {self.x: x_batch, self.y: y_batch, self.mask_with_labels: True}
                _, _, _, summary = self.sess.run([self.train_op,
                                                  self.mean_loss_op,
                                                  self.mean_accuracy_op,
                                                  self.merged_summary], feed_dict=feed_dict)
                loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                self.save_summary(summary, train_step + self.conf.reload_step, mode='train')
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
            else:
                x_batch, y_batch = self.data_reader.next_random_batch()
                feed_dict = {self.x: x_batch, self.y: y_batch, self.mask_with_labels: True}
                self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            if train_step % self.conf.VAL_FREQ == 0:
                self.evaluate(train_step)

            if train_step % self.conf.SAVE_FREQ == 0:
                self.save(train_step + self.conf.reload_step)

    def evaluate(self, train_step):
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_val_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_val, y_val = self.data_reader.next_batch(start, end)
            feed_dict = {self.x: x_val, self.y: y_val, self.mask_with_labels: False}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, train_step + self.conf.reload_step, mode='valid')
        print('-' * 30 + 'Validation' + '-' * 30)
        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}'.format(train_step,valid_loss, valid_acc))
        print('-' * 70)

    def test(self):
        pass

    def save(self, step):
        print('*' * 50)
        print('----> Saving the model at step #{0}'.format(step))
        print('*' * 50)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')

    def count_params(self):
        """Returns number of trainable parameters."""
        return count_parameters(self.sess)
