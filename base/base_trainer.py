import abc

from dotmap import DotMap

from base.base_data_loader import BaseDataLoader


class BaseTrainer:
    def __init__(self, data_loader: BaseDataLoader, config: DotMap) -> None:
        self.data_loader: BaseDataLoader = data_loader
        self.config: DotMap = config

    @abc.abstractmethod
    def train(self) -> None:
        raise NotImplementedError
