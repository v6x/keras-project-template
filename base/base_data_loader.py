import abc


class BaseDataLoader:
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def get_train_data_generator(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_validation_data_generator(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_data_generator(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_train_data_size(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_validation_data_size(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_data_size(self):
        raise NotImplementedError
