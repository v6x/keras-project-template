import os
import pickle

from base.base_model import BaseModel


# load optimizer weights only
class WithLoadOptimizerWeights(BaseModel):
    def __init__(self, model, model_name=None):
        super().__init__(model.config)
        self.model = model
        self.model_name = model_name

    def define_model(self, **kargs):
        return self.model.define_model(**kargs)

    def build_model(self, **kargs):
        compiled_model, parallel_compiled_model = self.model.build_model(**kargs)
        if self.config.trainer.epoch_to_continue > 0:
            # load optimizer weights if exists
            optimizer_checkpoint_path = f'{self.config.callbacks.checkpoint_dir}/{self.config.exp.name}-{self.config.trainer.epoch_to_continue:04d}-optimizer{"-" + self.model_name if self.model_name else ""}.pickle'
            if os.path.exists(optimizer_checkpoint_path):
                parallel_compiled_model._make_train_function()
                with open(optimizer_checkpoint_path, 'rb') as handle:
                    loaded_weight_values = pickle.load(handle)
                    parallel_compiled_model.optimizer.set_weights(loaded_weight_values)
                print(f'Load optimizer weights from {optimizer_checkpoint_path}.')
            else:
                print(f'optimizer weights not found: {optimizer_checkpoint_path}.')

        return compiled_model, parallel_compiled_model


# load model weights and optimizer weights
class WithLoadWeights(WithLoadOptimizerWeights):
    def __init__(self, model, model_name=None):
        super().__init__(model, model_name)

    def define_model(self, **kargs):
        return super().define_model(**kargs)

    def build_model(self, **kargs):
        compiled_model, parallel_compiled_model = super().build_model(**kargs)
        if self.config.trainer.epoch_to_continue > 0:
            # load model weights
            checkpoint_path = f'{self.config.callbacks.checkpoint_dir}/{self.config.exp.name}-{self.config.trainer.epoch_to_continue:04d}{"-" + self.model_name if self.model_name else ""}-weights.hdf5'

            if os.path.exists(checkpoint_path):
                print(f'Load model weights from {checkpoint_path}.')
                compiled_model.load_weights(checkpoint_path)
            else:
                raise FileNotFoundError(f'checkpoint file not found: {checkpoint_path}')

        return compiled_model, parallel_compiled_model
