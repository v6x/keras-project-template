import abc

from utils.multi_gpu_utils import multi_gpu_model


class BaseModel:
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def build_model(self, **kargs):
        raise NotImplementedError

    @abc.abstractmethod
    def define_model(self, **kargs):
        raise NotImplementedError

    def multi_gpu_model(self, model):
        gpus = self.config.trainer.gpus
        return multi_gpu_model(model, gpus=gpus) if gpus > 1 else model
