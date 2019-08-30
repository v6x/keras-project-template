import os
import pickle
from typing import Optional, Set, Dict

import numpy as np
import tensorflow as tf
from dotmap import DotMap
from keras import backend as K
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
from tensorflow.python.summary.writer.writer import FileWriter

from utils.telegram_noti import send_noti_to_telegram


class ScalarCollageTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                 write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None) -> None:
        super().__init__(log_dir, histogram_freq, batch_size, write_graph, write_grads, write_images, embeddings_freq,
                         embeddings_layer_names, embeddings_metadata)

        self.writers: Dict[str, FileWriter] = {}
        self.written_writer_keys: Set[str] = set()

    def _write_logs(self, logs: dict, index: int):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue

            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value

            # modified
            if '/' in name:
                *tag, writer_key = name.split('/')
                tag = '/'.join(tag)
                summary_value.tag = tag
                if writer_key not in self.writers:
                    self.writers[writer_key] = tf.summary.FileWriter(os.path.join(self.log_dir, writer_key))
                writer = self.writers[writer_key]
                writer.add_summary(summary, index)
                self.written_writer_keys.add(writer_key)
            else:
                summary_value.tag = name
                self.writer.add_summary(summary, index)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        self.written_writer_keys.clear()
        super().on_epoch_end(epoch, logs)

        self.writer.flush()
        for writer_key in self.written_writer_keys:
            self.writers[writer_key].flush()

    def on_train_end(self, _) -> None:
        super().on_train_end(_)
        for writer_key in self.writers:
            self.writers[writer_key].close()


class OptimizerSaver(Callback):
    def __init__(self, config: DotMap, model_name: str, verbose: int = 0) -> None:
        super().__init__()
        self.checkpoint_dir: str = config.exp.checkpoints_dir
        self.model_name: str = model_name
        self.num_epochs: int = config.trainer.num_epochs
        self.save_checkpoint_freq: int = config.trainer.save_checkpoint_freq
        self.keep_checkpoint_freq: int = config.trainer.keep_checkpoint_freq
        self.verbose: int = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if (epoch + 1) % self.save_checkpoint_freq != 0:
            return

        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)

        filename = os.path.join(self.checkpoint_dir,
                                f'{epoch + 1:04d}-optimizer-{self.model_name}.pickle')

        with open(filename, 'wb') as handle:
            pickle.dump(weight_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # delete last checkpoint
        last_epoch = epoch + 1 - self.save_checkpoint_freq
        if last_epoch % self.keep_checkpoint_freq != 0:
            last_filename = os.path.join(self.checkpoint_dir,
                                         f'{last_epoch:04d}-optimizer-{self.model_name}.pickle')
            if os.path.exists(last_filename):
                os.remove(last_filename)

        if self.verbose > 0:
            print('\nEpoch %05d: saving optimizer weights to %s' % (epoch + 1, filename))


class TrainProgressAlertCallback(Callback):
    def __init__(self, experiment_name: str, total_epochs: int) -> None:
        super().__init__()
        self.experiment_name: str = experiment_name
        self.total_epochs: int = total_epochs
        self.epoch: int = 0

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        # zero base to one base
        self.epoch = epoch + 1

    def on_train_begin(self, logs: Optional[int] = None) -> None:
        send_noti_to_telegram(f'{self.experiment_name} train started for {self.total_epochs} epochs')

    def on_train_end(self, logs: Optional[int] = None) -> None:
        send_noti_to_telegram(f'{self.experiment_name} train ended at {self.epoch}/{self.total_epochs} epochs')


class ModelCheckpointWithKeepFreq(ModelCheckpoint):
    def __init__(self, filepath: str, keep_checkpoint_freq: int = 10, save_checkpoint_freq: int = 1,
                 monitor: str = 'val_loss', verbose: int = 0, save_best_only: bool = False,
                 save_weights_only: bool = False, mode: str = 'auto', period: int = 1) -> None:
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
        self.keep_checkpoint_freq: int = keep_checkpoint_freq
        self.save_checkpoint_freq: int = save_checkpoint_freq

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        if (epoch + 1) % self.save_checkpoint_freq != 0:
            return

        super().on_epoch_end(epoch, logs)

        last_epoch = epoch + 1 - self.save_checkpoint_freq
        if last_epoch % self.keep_checkpoint_freq != 0:
            last_checkpoint_path = self.filepath.format(epoch=last_epoch, **logs)
            if os.path.exists(last_checkpoint_path):
                print('\nEpoch %05d: remove saved last checkpoint %s' % (last_epoch, last_checkpoint_path))
                os.remove(last_checkpoint_path)


class ModelSaver(Callback):
    def __init__(self, checkpoint_dir: str, keep_checkpoint_freq: int, model_name: str, num_epochs: int,
                 verbose: int = 0) -> None:
        super().__init__()
        self.checkpoint_dir: str = checkpoint_dir
        self.keep_checkpoint_freq: int = keep_checkpoint_freq
        self.model_name: str = model_name
        self.num_epochs: int = num_epochs
        self.verbose: int = verbose
        self.epoch: int = 0

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        # zero base to one base
        self.epoch = epoch + 1

        if self.epoch % self.keep_checkpoint_freq == 0:
            self.save_model(self.epoch)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        super().on_train_end(logs)
        if self.epoch == self.num_epochs:
            self.save_model(self.num_epochs)

    def save_model(self, epoch: int) -> None:
        filename = os.path.join(self.checkpoint_dir, f'{self.model_name}-{epoch:04d}.hdf5')
        self.model.save(filename, overwrite=True)
        if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (epoch, filename))
