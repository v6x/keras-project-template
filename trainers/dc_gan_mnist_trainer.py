import copy
import datetime
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from keras import backend as K
from keras.callbacks import LearningRateScheduler

from base.base_trainer import BaseTrain
from utils.callback import ModelCheckpointWithKeepFreq, OptimizerSaver, ModelSaver, TerminateOnAnyNaN, \
    TrainProgressAlertCallback, ScalarCollageTensorBoard
from utils.image import denormalize_image


class MnistTrainer(BaseTrain):
    def __init__(self, generator, discriminator, parallel_discriminator, combined_model, parallel_combined_model,
                 data_loader, config):
        super(MnistTrainer, self).__init__(data_loader, config)
        self.generator = generator
        self.discriminator = parallel_discriminator
        self.serial_discriminator = discriminator
        self.combined_model = parallel_combined_model
        self.serial_combined_model = combined_model

        self.model_callbacks = defaultdict(list)
        self.init_callbacks()

    def init_callbacks(self):
        # decay learning rate from the half point
        def lr_scheduler(lr, epoch, epochs):
            return lr if epoch <= epochs // 2 else (1 - (epoch - epochs // 2) / (epochs // 2 + 1)) * lr

        # learning rate decay
        for model_name in ['combined_model', 'discriminator']:
            self.model_callbacks[model_name].append(
                LearningRateScheduler(schedule=lambda epoch: lr_scheduler(self.config.model.generator.lr, epoch,
                                                                          self.config.trainer.num_epochs))
            )
        # model saver

        self.model_callbacks['serial_combined_model'].append(
            ModelCheckpointWithKeepFreq(filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                                              '%s-{epoch:04d}-%s.hdf5' % (
                                                                  self.config.exp.name, 'combined')),
                                        keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                                        save_best_only=False,
                                        save_weights_only=False,
                                        verbose=1)
        )
        self.model_callbacks['serial_combined_model'].append(
            ModelCheckpointWithKeepFreq(filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                                              '%s-{epoch:04d}-%s-weights.hdf5' % (
                                                                  self.config.exp.name, 'combined')),
                                        keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                                        save_best_only=False,
                                        save_weights_only=True,
                                        verbose=1)
        )

        # save optimizer weights
        for model_name in ['combined_model', 'discriminator']:
            self.model_callbacks[model_name].append(
                OptimizerSaver(self.config, model_name)
            )

        # save individual models
        for model_name in ['combined_model', 'discriminator']:
            self.model_callbacks[model_name].append(
                ModelSaver(checkpoint_dir=self.config.callbacks.checkpoint_dir,
                           experiment_name=self.config.exp.name,
                           num_epochs=self.config.trainer.num_epochs,
                           verbose=1)
            )

        # tensorboard callback
        self.model_callbacks['combined_model'].append(
            ScalarCollageTensorBoard(log_dir=self.config.callbacks.tensorboard_log_dir,
                                     batch_size=self.config.trainer.batch_size,
                                     write_images=True)
        )

        # stop if encounter nan loss
        self.model_callbacks['combined_model'].append(TerminateOnAnyNaN())

        # send notification to telegram channel
        self.model_callbacks['combined_model'].append(
            TrainProgressAlertCallback(experiment_name=self.config.exp.name,
                                       total_epochs=self.config.trainer.num_epochs)
        )

        epochs = self.config.trainer.num_epochs
        steps_per_epoch = self.data_loader.get_train_data_size() // self.config.trainer.batch_size
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            callbacks_metrics = copy.copy(model.metrics_names) if hasattr(model, 'metrics_names') else []
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.set_model(model)
                callback.set_params({
                    'batch_size': self.config.trainer.batch_size,
                    'epoch': epochs,
                    'steps': steps_per_epoch,
                    'samples': self.data_loader.get_train_data_size(),
                    'verbose': True,
                    'do_validation': False,
                    'metrics': callbacks_metrics,
                    'model_name': model_name
                })

    def train(self):
        train_data_generator = self.data_loader.get_train_data_generator()
        steps_per_epoch = self.data_loader.get_train_data_size() // self.config.trainer.batch_size
        assert steps_per_epoch > 0

        test_data_generator = self.data_loader.get_test_data_generator()
        test_data_size = self.data_loader.get_test_data_size()

        real_label = np.ones(shape=(self.config.trainer.batch_size,), dtype=np.int32)
        fake_label = np.zeros(shape=(self.config.trainer.batch_size,), dtype=np.int32)

        epochs = self.config.trainer.num_epochs
        start_time = datetime.datetime.now()

        self.on_train_end()
        # self.combined_model.fit()
        for epoch in range(epochs):
            self.on_epoch_begin(epoch, {})
            epoch_logs = defaultdict(float)
            for step in range(1, steps_per_epoch + 1):
                batch_logs = {'batch': step, 'size': self.config.trainer.batch_size}
                self.on_batch_begin(step, batch_logs)

                noise = np.random.normal(0, 1, (self.config.trainer.batch_size, 100))
                generated_images = self.generator.predict(noise)
                real_images = next(train_data_generator)
                d_real_metric_names = self.d_metric_name(True)
                d_fake_metric_names = self.d_metric_name(False)
                loss_fake_discriminator = self.discriminator.train_on_batch(
                    generated_images, fake_label)
                loss_real_discriminator = self.discriminator.train_on_batch(
                    real_images, real_label)

                g_metric_names = self.g_metric_name()
                loss_generator = self.combined_model.train_on_batch(noise, real_label)
                loss_generator = [loss_generator]

                assert len(g_metric_names) == len(loss_generator)
                assert len(d_fake_metric_names) == len(loss_fake_discriminator)
                assert len(d_real_metric_names) == len(loss_real_discriminator)

                metrics = [(g_metric_names, loss_generator),
                           (d_fake_metric_names, loss_fake_discriminator),
                           (d_real_metric_names, loss_real_discriminator)]

                for (metric_names, metric_values) in metrics:
                    for metric_name, metric_value in zip(metric_names, metric_values):
                        batch_logs[metric_name] = metric_value

                    # print
                    print_str = f"[Epoch {epoch + 1}/{epochs}] [Batch {step}/{steps_per_epoch}]"
                    deliminator = ' '
                    for metric_name, metric_value in zip(metric_names, metric_values):
                        if 'acc' in metric_name:
                            metric_value = metric_value * 100
                        epoch_logs[metric_name] += metric_value
                        if 'acc' in metric_name:
                            print_str += f"{deliminator}{metric_name}={metric_value:.1f}%"
                        elif 'loss' in metric_name:
                            print_str += f"{deliminator}{metric_name}={metric_value:.4f}"
                        else:
                            print_str += f"{deliminator}{metric_name}={metric_value}"
                        if deliminator == ' ':
                            deliminator = ',\t'

                    print_str += f", time: {datetime.datetime.now() - start_time}"
                    print(print_str, flush=True)

                    for metric_name, metric_value in zip(metric_names, metric_values):
                        epoch_logs[metric_name] = metric_value

                self.on_batch_end(step, batch_logs)

            for k in epoch_logs:
                epoch_logs[k] /= steps_per_epoch
            epoch_logs = dict(epoch_logs)

            epoch_logs['g/lr'] = K.get_value(self.combined_model.optimizer.lr)
            epoch_logs['d/lr'] = K.get_value(self.discriminator.optimizer.lr)

            self.on_epoch_end(epoch, epoch_logs)
            if (epoch + 1) % self.config.trainer.pred_rate == 0:
                self.sample_images(epoch, test_data_generator, test_data_size)

        self.on_train_end()

    @staticmethod
    def d_metric_name(is_real=True):
        type = "real" if is_real else "fake"
        d_metric_names = [f'd/loss_{type}']
        d_accuracy = [f'd/acc_{type}']
        return d_metric_names + d_accuracy

    @staticmethod
    def g_metric_name():
        g_metric_names = ['g/loss']
        return g_metric_names

    def needs_stop_training(self):
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            if model.stop_training:
                return True

        return False

    def sample_images(self, epoch, data_generator, data_size):
        output_dir = f"{self.config.callbacks.predicted_dir}/{epoch + 1}/"
        os.makedirs(output_dir, exist_ok=True)

        images = []
        for _ in range(data_size // self.config.trainer.batch_size):
            input_x = next(data_generator)
            noise = np.random.normal(0, 1, (self.config.trainer.batch_size, 100))

            generated_images = self.generator.predict(noise)

            for image in generated_images:
                image = np.squeeze(image, axis=-1)
                images.append(denormalize_image(image))

        save_batch_size = self.config.trainer.pred_save_batch_size
        for i in range(0, len(images), save_batch_size):
            concat_images = np.concatenate(images[i:i + save_batch_size], axis=0)
            Image.fromarray(concat_images).save(f"{output_dir}/{i // save_batch_size}.png")

    def on_batch_begin(self, batch, logs=None):
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs):
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            model.stop_training = False
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_end(logs)
