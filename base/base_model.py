import abc
from typing import Tuple

from dotmap import DotMap
from keras import Model
from keras.optimizers import Optimizer
from keras.utils import multi_gpu_model


class BaseModel:
    def __init__(self, config: DotMap) -> None:
        super().__init__()
        self.config: DotMap = config

    @abc.abstractmethod
    def define_model(self, **kargs) -> Model:
        raise NotImplementedError

    # returns tuple of serial model and parallel model
    @abc.abstractmethod
    def build_model(self, **kargs) -> Tuple[Model, Model]:
        raise NotImplementedError

    def multi_gpu_model(self, model: Model) -> Model:
        gpus = self.config.trainer.n_gpus
        return multi_gpu_model(model, gpus=gpus) if gpus > 1 else model

    def process_optimizer(self, optimizer: Optimizer) -> Optimizer:
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd
            optimizer = hvd.DistributedOptimizer(optimizer)
        return optimizer
