import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from base.base_data_loader import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        BaseDataLoader.__init__(self, config)
        train_data, test_data = mnist.load_data()

        train_x_data, train_labels = train_data
        test_x_data, test_labels = test_data

        train_labels = to_categorical(train_labels, num_classes=10, dtype=np.float64)
        test_labels = to_categorical(test_labels, num_classes=10, dtype=np.float64)

        self.train_data = train_data
        self.train_data_with_labels = (train_x_data, train_labels)
        self.test_data = test_data
        self.test_data_with_labels = (test_x_data, test_labels)
        self.train_data_size = len(train_data[0])
        self.test_data_size = len(test_data[0])

    def get_train_data(self):
        images = []
        while True:
            for (image, label) in zip(self.train_data[0], self.train_data[1]):
                images.append(image)
                if len(images) >= self.config.trainer.batch_size:
                    data = np.array(images) / 127.5 - 1
                    data = np.expand_dims(data, -1)
                    images.clear()
                    yield data

    def get_train_data_with_labels(self):
        images = []
        labels = []
        while True:
            for (image, label) in zip(self.train_data_with_labels[0], self.train_data_with_labels[1]):
                images.append(image)
                labels.append(label)
                if len(images) >= self.config.trainer.batch_size:
                    x_data = np.array(images) / 127.5 - 1
                    x_data = np.expand_dims(x_data, -1)
                    y_data = np.array(labels, dtype=np.float64)
                    images.clear()
                    labels.clear()
                    yield x_data, y_data

    def get_test_data(self):
        images = []
        while True:
            for (image, label) in zip(self.test_data[0], self.test_data[1]):
                images.append(image)
                if len(images) >= self.config.trainer.batch_size:
                    data = np.array(images) / 127.5 - 1
                    data = np.expand_dims(data, -1)
                    images.clear()
                    yield data

    def get_test_data_with_labels(self):
        images = []
        labels = []
        while True:
            for (image, label) in zip(self.test_data[0], self.test_data[1]):
                images.append(image)
                labels.append(label)
                if len(images) >= self.config.trainer.batch_size:
                    x_data = np.array(images) / 127.5 - 1
                    x_data = np.expand_dims(x_data, -1)
                    y_data = np.array(labels, dtype=np.float64)
                    images.clear()
                    labels.clear()
                    yield x_data, y_data

    def get_train_data_generator(self):
        return self.get_train_data()

    def get_train_data_generator_with_labels(self):
        return self.get_train_data_with_labels()

    def get_validation_data_generator(self):
        raise NotImplementedError

    def get_test_data_generator(self):
        return self.get_test_data()

    def get_test_data_generator_with_labels(self):
        return self.get_test_data_with_labels()

    def get_train_data_size(self):
        return self.train_data_size

    def get_validation_data_size(self):
        raise NotImplementedError

    def get_test_data_size(self):
        return self.test_data_size
