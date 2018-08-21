import tensorflow as tf
from config import args
import os
from utils import write_spec

if args.model == 'original_capsule':
    from models.Original_CapsNet import Orig_CapsNet as Model
elif args.model == 'matrix_capsule':
    from models.Matrix_Capsule_EM_routing import MatrixCapsNet as Model
elif args.model == 'vector_capsule':
    from models.Deep_CapsNet import CapsNet as Model


def main(_):
    if args.mode not in ['train', 'test']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train or test")
    else:
        model = Model(tf.Session(), args)
        if not os.path.exists(args.modeldir+args.run_name):
            os.makedirs(args.modeldir+args.run_name)
        if not os.path.exists(args.logdir+args.run_name):
            os.makedirs(args.logdir+args.run_name)
        if args.mode == 'train':
            write_spec(args)
            model.train()
        elif args.mode == 'test':
            model.test(args.step_num)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
    tf.app.run()
