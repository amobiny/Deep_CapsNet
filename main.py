import tensorflow as tf
from config import args
from models.Deep_CapsNet import CapsNet
from models.Original_CapsNet import Orig_CapsNet
from models.Fast_CapsNet import Fast_CapsNet_3D
from models.Matrix_Capsule_EM_routing import MatrixCapsNet

import os


def main(_):
    if args.mode not in ['train', 'test', 'predict']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train, test, or predict")
    else:
        # model = Fast_CapsNet_3D(tf.Session(), args)
        model = MatrixCapsNet(tf.Session(), args)
        # model.count_params()
        if not os.path.exists(args.modeldir+args.run_name):
            os.makedirs(args.modeldir+args.run_name)
        if not os.path.exists(args.logdir+args.run_name):
            os.makedirs(args.logdir+args.run_name)
        if not os.path.exists(args.savedir+args.run_name):
            os.makedirs(args.savedir+args.run_name)
        if args.mode == 'train':
            model.train()
        elif args.mode == 'test':
            model.test()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    tf.app.run()
