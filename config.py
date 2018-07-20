import tensorflow as tf
import time

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train or test')

# Training logs
flags.DEFINE_integer('max_step', 100000, '# of step for training (only for mnist)')
flags.DEFINE_integer('max_epoch', 1000, '# of step for training (only for nodule data)')
flags.DEFINE_integer('SAVE_FREQ', 1000, 'Number of steps to save model')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 500, 'Number of step to evaluate the network on Validation data')

# Hyper-parameters
flags.DEFINE_string('loss_type', 'spread', 'spread or margin')
flags.DEFINE_boolean('add_recon_loss', False, 'To add reconstruction loss')

# For margin loss
flags.DEFINE_float('m_plus', 0.9, 'm+ parameter')
flags.DEFINE_float('m_minus', 0.1, 'm- parameter')
flags.DEFINE_float('lambda_val', 0.5, 'Down-weighting parameter for the absent class')
# For reconstruction loss
flags.DEFINE_float('alpha', 0.0005, 'Regularization coefficient to scale down the reconstruction loss')
# For training
flags.DEFINE_integer('batch_size', 16, 'training batch size')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# data
flags.DEFINE_string('data', 'mnist', 'mnist or nodule or cifar10')
flags.DEFINE_integer('dim', 2, '2D or 3D for nodule data')
flags.DEFINE_boolean('one_hot', True, 'one-hot-encode the labels')
flags.DEFINE_boolean('data_augment', False, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 40, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('height', 28, 'Network input height size')
flags.DEFINE_integer('width', 28, 'Network input width size')
flags.DEFINE_integer('depth', 32, 'Network input depth size (in the case of 3D input images)')
flags.DEFINE_integer('channel', 1, 'Network input channel size')

# Directories
flags.DEFINE_string('run_name', 'orig', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Saved models directory')
flags.DEFINE_string('savedir', './Results/result/', 'Results saving directory')

flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')

# network architecture
flags.DEFINE_integer('num_cls', 10, 'Number of output classes')
flags.DEFINE_integer('prim_caps_dim', 8, 'Dimension of the PrimaryCaps in the Original_CapsNet')
flags.DEFINE_integer('digit_caps_dim', 16, 'Dimension of the DigitCaps in the Original_CapsNet')
flags.DEFINE_integer('h1', 512, 'Number of hidden units of the first FC layer of the reconstruction network')
flags.DEFINE_integer('h2', 1024, 'Number of hidden units of the second FC layer of the reconstruction network')

# Matrix Capsule architecture
flags.DEFINE_boolean('use_bias', True, 'Adds bias to init capsules')
flags.DEFINE_boolean('use_BN', True, 'Adds BN before conv1 layer')
flags.DEFINE_boolean('add_coords', True, 'Adds capsule coordinations')
flags.DEFINE_boolean('grad_clip', False, 'Adds gradient clipping to get rid of exploding gradient')
flags.DEFINE_boolean('L2_reg', False, 'Adds L2-regularization to all the network weights')
flags.DEFINE_float('lmbda', 5e-04, 'L2-regularization coefficient')
flags.DEFINE_boolean('add_decoder', True, 'Adds a fully connected decoder and reconstruction loss')
flags.DEFINE_integer('iter', 3, 'Number of EM-routing iterations')
flags.DEFINE_integer('A', 32, 'A in Figure 1 of the paper')
flags.DEFINE_integer('B', 32, 'B in Figure 1 of the paper')
flags.DEFINE_integer('C', 32, 'C in Figure 1 of the paper')
flags.DEFINE_integer('D', 32, 'D in Figure 1 of the paper')

args = tf.app.flags.FLAGS