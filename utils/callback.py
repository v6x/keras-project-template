import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint

from utils.telegram_noti import send_noti_to_telegram


class ScalarCollageTensorBoard(TensorBoard):

    def __init__(self, log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                 write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None):
        super().__init__(log_dir, histogram_freq, batch_size, write_graph, write_grads, write_images, embeddings_freq,
                         embeddings_layer_names, embeddings_metadata)

        self.writers = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if not self.validation_data and self.histogram_freq:
            raise ValueError('If printing histograms, validation_data must be '
                             'provided, and cannot be a generator.')
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        writer_keys = set()
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue

            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()

            # modified
            if '/' in name:
                writer_key, *tag = name.split('/')
                tag = '/'.join(tag)
                summary_value.tag = tag
                if writer_key not in self.writers:
                    self.writers[writer_key] = tf.summary.FileWriter(os.path.join(self.log_dir, writer_key))
                writer = self.writers[writer_key]
                writer.add_summary(summary, epoch)
                writer_keys.add(writer_key)
            else:
                summary_value.tag = name
                self.writer.add_summary(summary, epoch)

        self.writer.flush()
        for writer_key in writer_keys:
            self.writers[writer_key].flush()

    def on_train_end(self, _):
        super().on_train_end(_)
        for writer_key in self.writers:
            self.writers[writer_key].close()


class OptimizerSaver(Callback):
    def __init__(self, config, model_name, verbose=0):
        super().__init__()
        self.checkpoint_dir = config.callbacks.checkpoint_dir
        self.experiment_name = config.exp.name
        self.model_name = model_name
        self.num_epochs = config.trainer.num_epochs
        self.keep_checkpoint_freq = config.trainer.keep_checkpoint_freq
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)

        filename = os.path.join(self.checkpoint_dir,
                                f'{self.experiment_name}-{epoch + 1:04d}-optimizer-{self.model_name}.pickle')

        with open(filename, 'wb') as handle:
            pickle.dump(weight_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # delete last checkpoint
        if epoch % self.keep_checkpoint_freq != 0:
            last_filename = os.path.join(self.checkpoint_dir,
                                         f'{self.experiment_name}-{epoch:04d}-optimizer-{self.model_name}.pickle')
            if os.path.exists(last_filename):
                os.remove(last_filename)

        if self.verbose > 0:
            print('\nEpoch %05d: saving optimizer weights to %s' % (epoch + 1, filename))


class GeneratorSaver(Callback):
    def __init__(self, checkpoint_dir, experiment_name, num_epochs, verbose=0):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.verbose = verbose

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        g_xy_filename = os.path.join(self.checkpoint_dir,
                                     f'{self.experiment_name}-g_xy-{self.num_epochs:04d}.hdf5')
        self.model.save(g_xy_filename, overwrite=True)
        if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (self.num_epochs, g_xy_filename))


class TrainTimer(Callback):
    def __init__(self, verbose_list=('train', 'batch', 'epoch')):
        super().__init__()
        self.verbose_list = verbose_list
        self.datetime_batch_begin = datetime.now()
        self.datetime_batch_end = datetime.now()
        self.datetime_epoch_begin = datetime.now()
        self.datetime_epoch_end = datetime.now()
        self.datetime_train_begin = datetime.now()
        self.datetime_train_end = datetime.now()
        self.datetime_total_time_batch = timedelta(0)
        self.datetime_total_time_epoch = timedelta(0)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if 'train' in self.verbose_list:
            self.datetime_train_begin = datetime.now()

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if 'train' in self.verbose_list:
            self.datetime_train_end = datetime.now()
            print("time for train:", self.datetime_train_end - self.datetime_train_begin)
            print(f"time for mean_total_epochs({self.params['epochs']}):",
                  self.datetime_total_time_epoch / self.params['epochs'])
            print(f"time for mean_total_batches({self.params['steps']}):",
                  self.datetime_total_time_batch / self.params['steps'])

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs=logs)
        self.datetime_epoch_begin = datetime.now()
        self.datetime_total_time_batch = timedelta(0)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.datetime_epoch_end = datetime.now()
        timedelta_time_epoch = self.datetime_epoch_end - self.datetime_epoch_begin
        self.datetime_total_time_epoch += timedelta_time_epoch

        if 'epoch' in self.verbose_list:
            print(f"time for epoch({epoch}):", timedelta_time_epoch)
            print(f"time for mean_total_batches({self.params['steps']}):",
                  self.datetime_total_time_batch / self.params['steps'])

    def on_batch_begin(self, batch, logs=None):
        super().on_batch_begin(batch, logs)
        self.datetime_batch_begin = datetime.now()

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)
        self.datetime_batch_end = datetime.now()
        timedelta_time_batch = self.datetime_batch_end - self.datetime_batch_begin
        self.datetime_total_time_batch += timedelta_time_batch

        if 'batch' in self.verbose_list:
            print(f"time for batch({batch}):", timedelta_time_batch)


class TerminateOnAnyNaN(Callback):
    """
    Callback that terminates training when a NaN loss is encountered.
    """

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            if name.endswith('loss') and (np.isnan(value) or np.isinf(value)):
                print(f'Batch {batch}: Invalid loss, terminating training')
                self.model.stop_training = True


class TrainProgressAlertCallback(Callback):
    def __init__(self, experiment_name, total_epochs):
        super().__init__()
        self.experiment_name = experiment_name
        self.total_epochs = total_epochs
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        # zero base to one base
        self.epoch = epoch + 1

    def on_train_begin(self, logs=None):
        send_noti_to_telegram(f'{self.experiment_name} train started for {self.total_epochs} epochs')

    def on_train_end(self, logs=None):
        send_noti_to_telegram(f'{self.experiment_name} train ended at {self.epoch}/{self.total_epochs} epochs')


class ModelCheckpointWithKeepFreq(ModelCheckpoint):
    def __init__(self, filepath, keep_checkpoint_freq=10, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
        self.keep_checkpoint_freq = keep_checkpoint_freq

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        if epoch % self.keep_checkpoint_freq != 0:
            last_checkpoint_path = self.filepath.format(epoch=epoch, **logs)
            if os.path.exists(last_checkpoint_path):
                print('\nEpoch %05d: remove saved last checkpoint %s' % (epoch + 1, last_checkpoint_path))
                os.remove(last_checkpoint_path)


class ModelSaver(Callback):
    def __init__(self, checkpoint_dir, experiment_name, num_epochs, verbose=0):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        # zero base to one base
        self.epoch = epoch + 1

        # save every 10 epochs
        if self.epoch % 10 == 0:
            self.save_model(self.epoch)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if self.epoch == self.num_epochs:
            self.save_model(self.num_epochs)

    def save_model(self, epoch):
        filename = os.path.join(self.checkpoint_dir,
                                f'{self.experiment_name}-{self.params["model_name"]}-{epoch:04d}.hdf5')
        self.model.save(filename, overwrite=True)
        if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (epoch, filename))
