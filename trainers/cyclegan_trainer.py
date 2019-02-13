import copy
import datetime
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from tqdm import tqdm

from base.base_trainer import BaseTrain
from utils.callback import ScalarCollageTensorBoard, ModelCheckpointWithKeepFreq, OptimizerSaver, ModelSaver, \
    TerminateOnAnyNaN, TrainProgressAlertCallback
from utils.image import denormalize_image, read_image, normalize_image
from utils.pool import FakeImagePool


class CycleGanModelTrainer(BaseTrain):
    def __init__(self, g_xy, g_yx, d_x, parallel_d_x, d_y, parallel_d_y, combined_model, parallel_combined_model,
                 data_loader, config):
        super(CycleGanModelTrainer, self).__init__(data_loader, config)
        self.g_xy = g_xy
        self.g_yx = g_yx
        self.serial_d_x = d_x
        self.d_x = parallel_d_x
        self.serial_d_y = d_y
        self.d_y = parallel_d_y
        self.serial_combined = combined_model
        self.combined = parallel_combined_model

        self.model_callbacks = defaultdict(list)
        self.init_callbacks()

    def init_callbacks(self):
        def lr_scheduler(lr, epoch, epochs):
            return lr if epoch <= epochs // 2 else (1 - (epoch - epochs // 2) / (epochs // 2 + 1)) * lr

        # learning rate decay
        for model_name in ['combined', 'd_x', 'd_y']:
            self.model_callbacks[model_name].append(
                LearningRateScheduler(schedule=lambda epoch: lr_scheduler(self.config.model.generator.lr, epoch,
                                                                          self.config.trainer.num_epochs))
            )

        # model saver
        self.model_callbacks['serial_combined'].append(
            ModelCheckpointWithKeepFreq(filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                                              '%s-{epoch:04d}-%s.hdf5' % (
                                                                  self.config.exp.name, 'combined')),
                                        keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                                        save_best_only=False,
                                        save_weights_only=False,
                                        verbose=1)
        )
        self.model_callbacks['serial_combined'].append(
            ModelCheckpointWithKeepFreq(filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                                              '%s-{epoch:04d}-%s-weights.hdf5' % (
                                                                  self.config.exp.name, 'combined')),
                                        keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                                        save_best_only=False,
                                        save_weights_only=True,
                                        verbose=1)
        )

        # save optimizer weights
        for model_name in ['combined', 'd_x', 'd_y']:
            self.model_callbacks[model_name].append(
                OptimizerSaver(self.config, model_name)
            )

        # save individual models
        for model_name in ['g_xy', 'g_yx', 'd_x', 'd_y']:
            self.model_callbacks[model_name].append(
                ModelSaver(checkpoint_dir=self.config.callbacks.checkpoint_dir,
                           experiment_name=self.config.exp.name,
                           num_epochs=self.config.trainer.num_epochs,
                           verbose=1)
            )

        # tensorboard callback
        self.model_callbacks['combined'].append(
            ScalarCollageTensorBoard(log_dir=self.config.callbacks.tensorboard_log_dir,
                                     batch_size=self.config.trainer.batch_size,
                                     write_images=True)
        )

        # stop if encounter nan loss
        self.model_callbacks['combined'].append(TerminateOnAnyNaN())

        # send notification to telegram channel on train start and end
        self.model_callbacks['combined'].append(
            TrainProgressAlertCallback(experiment_name=self.config.exp.name,
                                       total_epochs=self.config.trainer.num_epochs)
        )

        # initialize callbacks by setting model and params
        epochs = self.config.trainer.num_epochs
        steps_per_epoch = self.data_loader.get_train_data_size() // self.config.trainer.batch_size
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            # not compiled model does not have a metrics_names attribute.
            callback_metrics = copy.copy(model.metrics_names) if hasattr(model, 'metrics_names') else []

            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.set_model(model)
                callback.set_params({
                    'batch_size': self.config.trainer.batch_size,
                    'epochs': epochs,
                    'steps': steps_per_epoch,
                    'samples': self.data_loader.get_train_data_size(),
                    'verbose': True,
                    'do_validation': False,
                    'metrics': callback_metrics,
                    'model_name': model_name
                })

    def train(self):
        train_data_generator = self.data_loader.get_train_data_generator()
        steps_per_epoch = self.data_loader.get_train_data_size() // self.config.trainer.batch_size
        assert steps_per_epoch > 0

        test_data_generator = self.data_loader.get_test_data_generator()
        test_data_size = self.data_loader.get_test_data_size()

        fake_x_pool = FakeImagePool(self.config.trainer.fake_pool_size)
        fake_y_pool = FakeImagePool(self.config.trainer.fake_pool_size)

        batch_shape = (self.config.trainer.batch_size,
                       self.config.dataset.image_size // 8, self.config.dataset.image_size // 8, 1)

        zeros = np.zeros(shape=batch_shape, dtype=np.float32)
        ones = np.ones(shape=batch_shape, dtype=np.float32)

        epochs = self.config.trainer.num_epochs
        start_time = datetime.datetime.now()

        self.on_train_begin()
        for epoch in range(self.config.trainer.epoch_to_continue, epochs):
            self.on_epoch_begin(epoch, {})

            epoch_logs = defaultdict(float)
            for step in range(1, steps_per_epoch + 1):
                batch_logs = {'batch': step, 'size': self.config.trainer.batch_size}
                self.on_batch_begin(step, batch_logs)

                imgs_x, imgs_y = next(train_data_generator)

                fakes_y = self.g_xy.predict(imgs_x)
                fakes_x = self.g_yx.predict(imgs_y)

                # train discriminator using history of fake images (Shrivastava et al)
                fakes_x = fake_x_pool.query(fakes_x)
                fakes_y = fake_y_pool.query(fakes_y)

                if self.config.trainer.label_smoothing:
                    zeros = np.random.uniform(0, 0.2, size=batch_shape)
                    ones = np.random.uniform(0.8, 1.0, size=batch_shape)

                # train discriminator
                dx_loss_real = self.d_x.train_on_batch(imgs_x, ones)  # real
                dx_loss_fake = self.d_x.train_on_batch(fakes_x, zeros)  # fake
                dx_loss = np.add(dx_loss_real, dx_loss_fake) / 2
                dy_loss_real = self.d_y.train_on_batch(imgs_y, ones)  # real
                dy_loss_fake = self.d_y.train_on_batch(fakes_y, zeros)  # fake
                dy_loss = np.add(dy_loss_real, dy_loss_fake) / 2
                d_loss = np.add(dx_loss, dy_loss) / 2

                # train generator
                g_loss = self.combined.train_on_batch([imgs_x, imgs_y], [ones, ones, imgs_x, imgs_y, imgs_x, imgs_y])

                metric_names = self.get_metric_names()
                metric_values = list(d_loss) + list(dx_loss_real) + list(dx_loss_fake) + \
                                list(dy_loss_real) + list(dy_loss_fake) + list(g_loss)

                assert len(metric_names) == len(metric_values)
                for metric_name, metric_value in zip(metric_names, metric_values):
                    batch_logs[metric_name] = metric_value
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

                self.on_batch_end(step, batch_logs)
                if self.needs_stop_training():
                    break
            if self.needs_stop_training():
                break

            # sum to average
            for k in epoch_logs:
                epoch_logs[k] /= steps_per_epoch
            epoch_logs = dict(epoch_logs)

            # additional log
            epoch_logs['g/lr'] = K.get_value(self.combined.optimizer.lr)
            epoch_logs['dx/lr'] = K.get_value(self.d_x.optimizer.lr)
            epoch_logs['dy/lr'] = K.get_value(self.d_y.optimizer.lr)

            self.on_epoch_end(epoch, epoch_logs)
            if (epoch + 1) % self.config.trainer.pred_rate == 0:
                self.sample_images(epoch, test_data_generator, test_data_size)
        self.on_train_end()

    @staticmethod
    def get_metric_names():
        d_metric_names = ['d/loss', 'd/acc']
        d_x_real_metric_names = ['dx real/loss', 'dx real/acc']
        d_x_fake_metric_names = ['dx fake/loss', 'dx fake/acc']
        d_y_real_metric_names = ['dy real/loss', 'dy real/acc']
        d_y_fake_metric_names = ['dy fake/loss', 'dy fake/acc']
        g_metric_names = ['g/loss',
                          'g adv x/loss', 'g adv y/acc',
                          'g recons x/loss', 'g recons y/acc',
                          'g identity x/loss', 'g identity y/acc']
        return d_metric_names + d_x_real_metric_names + d_x_fake_metric_names + \
               d_y_real_metric_names + d_y_fake_metric_names + g_metric_names

    def predict(self, input_filename):
        img_x = read_image(input_filename)
        # img_x = resize_image(img_x, (256, 256))
        img_x = normalize_image(np.expand_dims(img_x, axis=0))
        fake_y = self.g_xy.predict(img_x)
        fake_y = denormalize_image(fake_y)
        Image.fromarray(np.concatenate(fake_y, axis=0)).save(f"fake.png")

    def predict_all(self, max_num, original_size=False):
        output_dir = f"{self.config.callbacks.predicted_dir}/all/{self.config.trainer.epoch_to_continue}/"
        os.makedirs(output_dir, exist_ok=True)

        if original_size:
            data_generator = self.data_loader.get_original_test_data_generator([1 / 2])
            data_size = self.data_loader.get_original_test_data_size()
            for i in tqdm(range(min(max_num, data_size))):
                img_x, _ = next(data_generator)
                fake_y = self.g_xy.predict(np.expand_dims(img_x, axis=0))[0]
                recons_x = self.g_yx.predict(np.expand_dims(fake_y, axis=0))[0]

                image = np.concatenate([img_x, fake_y, recons_x], axis=1)
                Image.fromarray(denormalize_image(image)).save(f"{output_dir}/{i}.png")
        else:
            data_generator = self.data_loader.get_test_data_generator()
            data_size = self.data_loader.get_test_data_size()
            images = []
            for _ in range(min(max_num, data_size) // self.config.trainer.batch_size):
                imgs_x, imgs_y = next(data_generator)

                fake_y = self.g_xy.predict(imgs_x)
                fake_x = self.g_yx.predict(imgs_y)

                recons_x = self.g_yx.predict(fake_y)
                recons_y = self.g_xy.predict(fake_x)

                for i in range(imgs_x.shape[0]):
                    image = np.concatenate([imgs_x[i], fake_y[i], recons_x[i], imgs_y[i], fake_x[i], recons_y[i]],
                                           axis=1)
                    images.append(denormalize_image(image))

            save_batch_size = self.config.trainer.pred_save_batch_size
            for i in range(0, len(images), save_batch_size):
                concat_images = np.concatenate(images[i:i + save_batch_size], axis=0)
                Image.fromarray(concat_images).save(f"{output_dir}/{i // save_batch_size}.png")

    def sample_images(self, epoch, data_generator, data_size):
        output_dir = f"{self.config.callbacks.predicted_dir}/{epoch+1}/"
        os.makedirs(output_dir, exist_ok=True)

        images = []
        for _ in range(data_size // self.config.trainer.batch_size):
            imgs_x, imgs_y = next(data_generator)

            fake_y = self.g_xy.predict(imgs_x)
            fake_x = self.g_yx.predict(imgs_y)

            recons_x = self.g_yx.predict(fake_y)
            recons_y = self.g_xy.predict(fake_x)

            for i in range(imgs_x.shape[0]):
                image = np.concatenate([imgs_x[i], fake_y[i], recons_x[i], imgs_y[i], fake_x[i], recons_y[i]], axis=1)
                images.append(denormalize_image(image))

        save_batch_size = self.config.trainer.pred_save_batch_size
        for i in range(0, len(images), save_batch_size):
            concat_images = np.concatenate(images[i:i + save_batch_size], axis=0)
            Image.fromarray(concat_images).save(f"{output_dir}/{i // save_batch_size}.png")

    def needs_stop_training(self):
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            if model.stop_training:
                return True

        return False

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
