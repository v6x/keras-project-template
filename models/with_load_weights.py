import os
import pickle
from typing import Optional, Tuple

from keras import Model

from base.base_model import BaseModel


# load optimizer weights only
class WithLoadOptimizerWeights(BaseModel):
    def __init__(self, model: BaseModel, model_name: Optional[str] = None) -> None:
        super().__init__(model.config)
        self.model: BaseModel = model
        self.model_name: Optional[str] = model_name

    def define_model(self, **kwargs) -> Model:
        return self.model.define_model(**kwargs)

    def build_model(self, **kwargs) -> Tuple[Model, Model]:
        compiled_model, parallel_compiled_model = self.model.build_model(**kwargs)
        # Load optimizer only when training, NOT for fine tuning
        if not self.config.exp.fine_tuning_phase and self.config.trainer.epoch_to_continue > 0:
            # load optimizer weights if exists
            optimizer_checkpoint_path = f'{self.config.exp.checkpoints_dir}/{self.config.trainer.epoch_to_continue:04d}' \
                f'-optimizer{"-" + self.model_name if self.model_name else ""}.pickle'
            if os.path.exists(optimizer_checkpoint_path):
                parallel_compiled_model._make_train_function()
                with open(optimizer_checkpoint_path, "rb") as handle:
                    loaded_weight_values = pickle.load(handle)
                    parallel_compiled_model.optimizer.set_weights(loaded_weight_values)
                print(f"Load optimizer weights from {optimizer_checkpoint_path}.")
            else:
                print(f"optimizer weights not found: {optimizer_checkpoint_path}.")

        return compiled_model, parallel_compiled_model


# load model weights and optimizer weights
class WithLoadWeights(WithLoadOptimizerWeights):
    def __init__(self, model: BaseModel, model_name: Optional[str] = None) -> None:
        super().__init__(model, model_name)

    def define_model(self, **kargs) -> Model:
        return super().define_model(**kargs)

    def build_model(self, **kargs) -> Tuple[Model, Model]:
        compiled_model, parallel_compiled_model = super().build_model(**kargs)
        if self.config.trainer.epoch_to_continue > 0:
            checkpoint_path = f'{self.config.exp.checkpoints_dir}/{self.config.trainer.epoch_to_continue:04d}' \
                f'{"-" + self.model_name if self.model_name else ""}.hdf5'
            if os.path.exists(checkpoint_path):
                compiled_model.load_weights(checkpoint_path)
                print(f"Load model weights from {checkpoint_path}.")
            else:
                raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")

        return compiled_model, parallel_compiled_model
