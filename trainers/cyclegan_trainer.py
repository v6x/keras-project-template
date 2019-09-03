import datetime
import os
from collections import defaultdict
from typing import Optional, Generator

import numpy as np
from PIL import Image
from dotmap import DotMap
from keras import backend as K, Model
from keras.callbacks import LearningRateScheduler
from tqdm import tqdm

from base.base_data_loader import BaseDataLoader
from base.base_trainer import BaseTrainer
from utils.callback import ScalarCollageTensorBoard, ModelCheckpointWithKeepFreq, OptimizerSaver, ModelSaver, \
    TrainProgressAlertCallback
from utils.image import denormalize_image, read_image, normalize_image
from utils.pool import FakeImagePool


class CycleGanModelTrainer(BaseTrainer):
    def __init__(self, g_xy: Model, g_yx: Model, d_x: Model, parallel_d_x: Model, d_y: Model, parallel_d_y: Model,
                 combined_model: Model, parallel_combined_model: Model,
                 data_loader: BaseDataLoader, config: DotMap) -> None:
        super().__init__(data_loader, config)
        # serial models are subject to saving model and parallel models are subject to trainig model
        self.g_xy: Model = g_xy
        self.g_yx: Model = g_yx
        self.serial_d_x: Model = d_x
        self.d_x: Model = parallel_d_x
        self.serial_d_y: Model = d_y
        self.d_y: Model = parallel_d_y
        self.serial_combined: Model = combined_model
        self.combined: Model = parallel_combined_model

        print(f"[*] # params of generator: {self.g_xy.count_params():,}")
        print(f"[*] # params of discriminator: {self.d_x.count_params():,}")

        self.model_callbacks: dict = defaultdict(list)
        self.init_callbacks()

    def init_callbacks(self) -> None:
        if self.config.trainer.use_lr_decay:
            # linear decay from the half of max_epochs
            def lr_scheduler(lr, epoch, max_epochs):
                return min(lr, 2 * lr * (1 - epoch / max_epochs))

            self.model_callbacks["combined"].append(
                LearningRateScheduler(schedule=lambda epoch: lr_scheduler(self.config.model.generator.lr, epoch,
                                                                          self.config.trainer.num_epochs)))
            for model_name in ['d_x', 'd_y']:
                self.model_callbacks[model_name].append(
                    LearningRateScheduler(schedule=lambda epoch: lr_scheduler(self.config.model.discriminator.lr, epoch,
                                                                              self.config.trainer.num_epochs)))
        # if horovod used, only worker 0 saves checkpoints
        is_master = True
        is_local_master = True
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd

            is_master = hvd.rank() == 0
            is_local_master = hvd.local_rank() == 0

        # horovod callbacks
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd

            self.model_callbacks["combined"].append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            self.model_callbacks["combined"].append(hvd.callbacks.MetricAverageCallback())
            self.model_callbacks["combined"].append(
                hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))

        if is_local_master:
            # model saver
            self.model_callbacks["serial_combined"].append(
                ModelCheckpointWithKeepFreq(
                    filepath=os.path.join(self.config.exp.checkpoints_dir, "{epoch:04d}-combined.hdf5"),
                    keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                    save_checkpoint_freq=self.config.trainer.save_checkpoint_freq,
                    save_best_only=False,
                    save_weights_only=True,
                    verbose=1))

            # save optimizer weights
            for model_name in ['combined', 'd_x', 'd_y']:
                self.model_callbacks[model_name].append(OptimizerSaver(self.config, model_name))
        if is_master:
            # save individual models
            for model_name in ['g_xy', 'g_yx', 'd_x', 'd_y']:
                self.model_callbacks[model_name].append(
                    ModelSaver(
                        checkpoint_dir=self.config.exp.checkpoints_dir,
                        keep_checkpoint_freq=self.config.trainer.keep_checkpoint_freq,
                        model_name=model_name,
                        num_epochs=self.config.trainer.num_epochs,
                        verbose=1))

            # send notification to telegram channel on train start and end
            self.model_callbacks["combined"].append(TrainProgressAlertCallback(experiment_name=self.config.exp.name,
                                                                               total_epochs=self.config.trainer.num_epochs))

            # tensorboard callback
            self.model_callbacks["combined"].append(
                ScalarCollageTensorBoard(log_dir=self.config.exp.tensorboard_dir,
                                         batch_size=self.config.trainer.batch_size,
                                         write_images=True))

        # initialize callbacks by setting model and params
        epochs = self.config.trainer.num_epochs
        steps_per_epoch = self.data_loader.get_train_data_size() // self.config.trainer.batch_size
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")

            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.set_model(model)
                callback.set_params({
                    "batch_size": self.config.trainer.batch_size,
                    "epochs": epochs,
                    "steps": steps_per_epoch,
                    "samples": self.data_loader.get_train_data_size(),
                    "verbose": True,
                    "do_validation": False,
                    "model_name": model_name,
                })

    @staticmethod
    def d_metric_names(model_name):
        return [f"loss/D_{model_name}", f"accuracy/D_{model_name}"]

    @staticmethod
    def g_metric_names():
        return ["loss/G", "loss/G_x_adv", "loss/G_y_adv", "loss/G_x_recon", "loss/G_y_recon",
                "loss/G_x_identity", "loss/G_y_identity"]
    
    def train(self):
        train_data_generator = self.data_loader.get_train_data_generator()
        batch_size = self.config.trainer.batch_size

        steps_per_epoch = self.data_loader.get_train_data_size() // batch_size
        if self.config.trainer.use_horovod:
            import horovod.keras as hvd

            steps_per_epoch //= hvd.size()
        assert steps_per_epoch > 0

        valid_data_generator = self.data_loader.get_validation_data_generator()
        valid_data_size = self.data_loader.get_validation_data_size()

        fake_x_pool = FakeImagePool(self.config.trainer.fake_pool_size)
        fake_y_pool = FakeImagePool(self.config.trainer.fake_pool_size)

        batch_shape = (self.config.trainer.batch_size,
                       self.config.dataset.image_size // 8, self.config.dataset.image_size // 8, 1)

        fake = np.zeros(shape=batch_shape, dtype=np.float32)
        real = np.ones(shape=batch_shape, dtype=np.float32)

        epochs = self.config.trainer.num_epochs
        start_time = datetime.datetime.now()

        self.on_train_begin()
        for epoch in range(self.config.trainer.epoch_to_continue, epochs):
            self.on_epoch_begin(epoch, {})

            epoch_logs = defaultdict(float)
            for step in range(1, steps_per_epoch + 1):
                batch_logs = {'batch': step, 'size': self.config.trainer.batch_size}
                batch_logs = {"batch": step, "size": self.config.trainer.batch_size}
                self.on_batch_begin(step, batch_logs)

                imgs_x, imgs_y = next(train_data_generator)

                fakes_y = self.g_xy.predict(imgs_x)
                fakes_x = self.g_yx.predict(imgs_y)

                # train discriminator using history of fake images (Shrivastava et al)
                fakes_x = fake_x_pool.query(fakes_x)
                fakes_y = fake_y_pool.query(fakes_y)

                if self.config.trainer.label_smoothing:
                    fake = np.random.uniform(0, 0.2, size=batch_shape)
                    real = np.random.uniform(0.8, 1.0, size=batch_shape)

                # train discriminator
                dx_loss_real = self.d_x.train_on_batch(imgs_x, real)
                dx_loss_fake = self.d_x.train_on_batch(fakes_x, fake)
                dy_loss_real = self.d_y.train_on_batch(imgs_y, real)
                dy_loss_fake = self.d_y.train_on_batch(fakes_y, fake)

                # train generator
                g_loss = self.combined.train_on_batch([imgs_x, imgs_y], [real, real, imgs_x, imgs_y, imgs_x, imgs_y])

                dx_metric_names = self.d_metric_names("x")
                dy_metric_names = self.d_metric_names("y")
                g_metric_names = self.g_metric_names()

                assert len(dx_metric_names) == len(dx_loss_real) == len(dx_loss_fake)
                assert len(dy_metric_names) == len(dy_loss_real) == len(dy_loss_fake)
                assert len(g_metric_names) == len(g_loss)

                metric_logs = {}
                for metric_name, metric_value in zip(dx_metric_names + dy_metric_names,
                                                     dx_loss_real + dy_loss_real):
                    metric_logs[f"train/{metric_name}_real"] = \
                        metric_value * (100 if "accuracy" in metric_name.lower() else 1)

                for metric_name, metric_value in zip(dx_metric_names + dy_metric_names,
                                                     dx_loss_fake + dy_loss_fake):
                    metric_logs[f"train/{metric_name}_fake"] = \
                        metric_value * (100 if "accuracy" in metric_name.lower() else 1)

                for metric_name, metric_value in zip(g_metric_names, g_loss):
                    metric_logs[f"train/{metric_name}"] = metric_value

                batch_logs.update(metric_logs)
                for metric_name in metric_logs.keys():
                    if metric_name in epoch_logs:
                        epoch_logs[metric_name] += metric_logs[metric_name]
                    else:
                        epoch_logs[metric_name] = metric_logs[metric_name]

                print_str = f"[Epoch {epoch + 1}/{epochs}] [Batch {step}/{steps_per_epoch}]"
                deliminator = ' '
                for metric_name, metric_value in metric_logs.items():
                    if 'accuracy' in metric_name:
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

            # sum to average
            for k in epoch_logs:
                epoch_logs[k] /= steps_per_epoch
            epoch_logs = dict(epoch_logs)

            # additional log
            epoch_logs['train/lr/G'] = K.get_value(self.combined.optimizer.lr)
            epoch_logs['train/lr/D_x'] = K.get_value(self.d_x.optimizer.lr)
            epoch_logs['train/lr/D_y'] = K.get_value(self.d_y.optimizer.lr)

            self.on_epoch_end(epoch, epoch_logs)
            if (epoch + 1) % self.config.trainer.predict_freq == 0:
                self.sample_valid_images(epoch, valid_data_generator, valid_data_size)

        self.predict_test_images(epochs)
        self.on_train_end()

    def sample_valid_images(self, epoch: int, data_generator: Generator, data_size: int) -> None:
        output_dir = f"{self.config.exp.sample_dir}/{epoch + 1}/"
        self.sample_images(data_generator, data_size, output_dir)

    def predict_test_images(self, epochs: int) -> None:
        data_generator = self.data_loader.get_test_data_generator()
        data_size = self.data_loader.get_test_data_size()
        output_dir = f"{self.config.exp.sample_dir}/all/{epochs}/"
        self.sample_images(data_generator, data_size, output_dir)

    def sample_images(self, data_generator: Generator, data_size: int, output_dir: str) -> None:
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

        save_batch_size = self.config.trainer.batch_size
        for i in range(0, len(images), save_batch_size):
            concat_images = np.concatenate(images[i:i + save_batch_size], axis=0)
            Image.fromarray(concat_images).save(f"{output_dir}/{i // save_batch_size}.png")

    def on_batch_begin(self, batch: int, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict]) -> None:
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[dict]) -> None:
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            model.stop_training = False
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_end(logs)
