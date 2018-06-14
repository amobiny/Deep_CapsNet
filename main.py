import tensorflow as tf
from config import args
from models.CapsNet import CapsNet

import os


def main(_):
    if args.mode not in ['train', 'test', 'predict']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train, test, or predict")
    else:
        model = CapsNet(tf.Session(), args)
        # model.count_params()
        if not os.path.exists(args.modeldir):
            os.makedirs(args.modeldir)
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        if args.mode == 'train':
            model.train()
        elif args.mode == 'test':
            model.test()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    tf.app.run()