from tensorflow.examples.tutorials.mnist import input_data


class DataLoader(object):

    def __init__(self, cfg):
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.batch_size = cfg.batch_size
        self.num_tr = cfg.num_tr
        self.mnist = input_data.read_data_sets("data/mnist", one_hot=True)
        self.height, self.width, self.depth = cfg.height, cfg.width, cfg.depth
        self.x_train, self.y_train = self.mnist.train.images, self.mnist.train.labels

    def next_batch(self):

        if self.augment:
            pass
        return x, y

    def get_validation(self):
        x_valid, y_valid = self.mnist.validation.images, self.mnist.validation.labels
        return x_valid, y_valid