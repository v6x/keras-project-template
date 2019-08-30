import abc
from typing import Generator

from dotmap import DotMap


class BaseDataLoader:
    def __init__(self, config: DotMap) -> None:
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def get_train_data_generator(self) -> Generator:
        raise NotImplementedError

    @abc.abstractmethod
    def get_validation_data_generator(self) -> Generator:
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_data_generator(self) -> Generator:
        raise NotImplementedError

    @abc.abstractmethod
    def get_train_data_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_validation_data_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_test_data_size(self) -> int:
        raise NotImplementedError
