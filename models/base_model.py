import tensorflow as tf
import os
import numpy as np


class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.is_training = True
        self.input_shape = [None, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [None, self.conf.num_cls]
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.float32, self.output_shape, name='annotation')
            # self.is_training = tf.placeholder_with_default(True, shape=(), name='is_training')
            self.mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

    def loss_func(self):
        with tf.name_scope('Loss'):
            y_one_hot = tf.one_hot(self.y, depth=self.conf.num_cls, axis=4, name='y_one_hot')
            if self.conf.loss_type == 'cross-entropy':
                with tf.name_scope('cross_entropy'):
                    loss = cross_entropy(y_one_hot, self.logits, self.conf.nu_cls)
            elif self.conf.loss_type == 'dice':
                with tf.name_scope('dice_coefficient'):
                    loss = dice_coeff(y_one_hot, self.logits)
            with tf.name_scope('L2_loss'):
                l2_loss = tf.reduce_sum(
                    self.conf.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('reg_weights')]))
            with tf.name_scope('total'):
                self.loss = loss + l2_loss

    def accuracy_func(self):
        with tf.name_scope('Accuracy'):
            self.y_pred = tf.argmax(self.logits, axis=4, name='decode_pred')
            correct_prediction = tf.equal(self.y, self.y_pred, name='correct_pred')
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        steps_per_epoch = self.conf.N // self.conf.batch_size
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   global_step,
                                                   steps_per_epoch,
                                                   0.97,
                                                   staircase=True)
        self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        summary_list = [tf.summary.scalar('learning_rate', self.learning_rate),
                        tf.summary.scalar('loss', self.loss),
                        tf.summary.scalar('accuracy', self.accuracy),
                        tf.summary.image('train/original_image',
                                         self.x[:, :, :, self.conf.depth / 2],
                                         max_outputs=self.conf.batch_size),
                        tf.summary.image('train/prediction_mask',
                                         tf.cast(tf.expand_dims(self.y_pred[:, :, :, self.conf.depth / 2], -1),
                                                 tf.float32),
                                         max_outputs=self.conf.batch_size),
                        tf.summary.image('train/original_mask',
                                         tf.cast(tf.expand_dims(self.y[:, :, :, self.conf.depth / 2], -1), tf.float32),
                                         max_outputs=self.conf.batch_size)]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step):
        print('----> Summarizing at step {}'.format(step))
        if self.is_training:
            self.train_writer.add_summary(summary, step)
        else:
            self.valid_writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
        else:
            print('----> Start Training')
        data_reader = DataLoader(self.conf)
        for train_step in range(1, self.conf.max_step + 1):
            print('Step: {}'.format(train_step))
            self.is_training = True
            if train_step % self.conf.SUMMARY_FREQ == 0:
                x_batch, y_batch = data_reader.next_batch()
                feed_dict = {self.x: x_batch, self.y: y_batch, self.keep_prob: 0.7}
                _, loss, acc, summary = self.sess.run([self.train_op, self.loss, self.accuracy, self.merged_summary],
                                                      feed_dict=feed_dict)
                self.save_summary(summary, train_step + self.conf.reload_step)
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
            else:
                x_batch, y_batch = data_reader.next_batch()
                feed_dict = {self.x: x_batch, self.y: y_batch, self.keep_prob: 0.7}
                self.sess.run(self.train_op, feed_dict=feed_dict)
            if train_step % self.conf.VAL_FREQ == 0:
                self.is_training = False
                x_val, y_val = data_reader.get_validation()
                feed_dict = {self.x: x_val, self.y: y_val, self.keep_prob: 1}
                loss, acc, summary = self.sess.run([self.loss, self.accuracy, self.merged_summary], feed_dict=feed_dict)
                self.save_summary(summary, train_step + self.conf.reload_step)
                print('-' * 30 + 'Validation' + '-' * 30)
                print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}'.format(train_step, loss, acc))
                print('-' * 70)
            if train_step % self.conf.SAVE_FREQ == 0:
                self.save(train_step + self.conf.reload_step)

    def test(self):
        pass

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
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