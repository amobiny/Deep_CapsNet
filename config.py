import tensorflow as tf
import time

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train or test')

# Training logs
flags.DEFINE_integer('max_step', 250000, '# of step for training')
flags.DEFINE_integer('SAVE_FREQ', 1000, 'Number of steps to save model')
flags.DEFINE_integer('SUMMARY_FREQ', 10, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 10, 'Number of step to evaluate the network on Validation data')

# Hyper-parameters
# For margin loss
flags.DEFINE_float('m_plus', 0.9, 'm+ parameter')
flags.DEFINE_float('m_minus', 0.1, 'm- parameter')
flags.DEFINE_float('lambda_val', 0.5, 'Down-weighting parameter for the absent class')
# For reconstruction loss
flags.DEFINE_float('--alpha', 0.0005, 'Regularization coefficient to scale down the reconstruction loss')
# For training
flags.DEFINE_integer('batch_size', 16, 'training batch size')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# data
flags.DEFINE_string('train_data_dir', './data/train_data/', 'Training data directory')
flags.DEFINE_string('valid_data_dir', './data/valid_data/', 'Validation data directory')
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 40, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('num_tr', 55000, 'Total number of training images')
flags.DEFINE_integer('height', 28, 'Network input height size')
flags.DEFINE_integer('width', 28, 'Network input width size')
flags.DEFINE_integer('depth', 28, 'Network input depth size')
flags.DEFINE_integer('channel', 1, 'Network input channel size')

# Directories
flags.DEFINE_string('logdir', './log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './model_dir', 'Model directory')
flags.DEFINE_string('savedir', './result', 'Result saving directory')

flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')

# network architecture
flags.DEFINE_integer('num_cls', 10, 'Number of output classes')
flags.DEFINE_integer('digit_caps_dim', 16, 'Dimension of the DigitCaps')
flags.DEFINE_integer('h1', 512, 'Number of hidden units of the first FC layer of the reconstruction network')
flags.DEFINE_integer('h2', 1024, 'Number of hidden units of the second FC layer of the reconstruction network')

args = tf.app.flags.FLAGS