from glob import glob

import numpy as np
from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ, BatchData, LocallyShuffleData, LMDBDataPoint

from base.base_data_loader import BaseDataLoader
from utils.dataflow import GeneratorToDataFlow, InfiniteDataFlow
from utils.image import read_image, resize_image_by_ratio


class ProcessingDataFlow(RNGDataFlow):
    def __init__(self, ds, crop_size=None, fit_resize=True, random_fit=True, random_fit_max_ratio=1.3,
                 random_flip=False,
                 random_brightness=False, random_contrast=False):
        self.ds = ds
        self.crop_size = crop_size
        self.fit_resize = fit_resize
        self.random_fit_max_ratio = random_fit_max_ratio
        self.random_fit = random_fit
        self.random_flip = random_flip
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast

    def reset_state(self):
        super().reset_state()
        self.ds.reset_state()

    def get_data(self):
        for path_a, path_b in self.ds.get_data():
            img_a = read_image(path_a)
            img_b = read_image(path_b)

            if self.fit_resize:
                min_ratio_a = max(self.crop_size[0] / img_a.shape[0], self.crop_size[1] / img_a.shape[1])
                ratio_a = np.random.uniform(min_ratio_a,
                                            min_ratio_a * self.random_fit_max_ratio) if self.random_fit else min_ratio_a
                img_a = resize_image_by_ratio(img_a, ratio_a)

                min_ratio_b = max(self.crop_size[0] / img_b.shape[0], self.crop_size[1] / img_b.shape[1])
                ratio_b = np.random.uniform(min_ratio_b,
                                            min_ratio_b * self.random_fit_max_ratio) if self.random_fit else min_ratio_b
                img_b = resize_image_by_ratio(img_b, ratio_b)

            if self.crop_size is not None:
                w_offset_a = np.random.randint(0, img_a.shape[1] - self.crop_size[1] + 1)
                h_offset_a = np.random.randint(0, img_a.shape[0] - self.crop_size[0] + 1)
                img_a = img_a[h_offset_a:h_offset_a + self.crop_size[0], w_offset_a:w_offset_a + self.crop_size[1], :]
                assert img_a.shape[:2] == self.crop_size

                w_offset_b = np.random.randint(0, img_b.shape[1] - self.crop_size[1] + 1)
                h_offset_b = np.random.randint(0, img_b.shape[0] - self.crop_size[0] + 1)
                img_b = img_b[h_offset_b:h_offset_b + self.crop_size[0], w_offset_b:w_offset_b + self.crop_size[1], :]
                assert img_b.shape[:2] == self.crop_size

            if self.random_flip and np.random.random() > 0.5:
                img_a = np.fliplr(img_a)
                img_b = np.fliplr(img_b)

            img_a = img_a.astype(np.float32)
            img_b = img_b.astype(np.float32)

            if self.random_brightness:
                img_a = np.clip(img_a + 255 * np.random.uniform(-0.05, 0.05), 0, 255)
                img_b = np.clip(img_b + 255 * np.random.uniform(-0.05, 0.05), 0, 255)

            if self.random_contrast:
                img_a = np.clip((img_a - 127.5) * np.random.uniform(0.9, 1.1) + 127.5, 0, 255)
                img_b = np.clip((img_b - 127.5) * np.random.uniform(0.9, 1.1) + 127.5, 0, 255)

            # normalize
            yield img_a / 127.5 - 1, img_b / 127.5 - 1


def paths_to_generator(paths_a, paths_b, shuffle):
    while True:
        if shuffle:
            np.random.shuffle(paths_a)
            np.random.shuffle(paths_b)

        for path_a, path_b in zip(paths_a, paths_b):
            yield path_a, path_b


class CycleGanDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

        if config.dataset.data_loader.use_lmdb:
            train_dataset = LMDBDataPoint(f'./datasets/{self.config.dataset.name}/train.lmdb', shuffle=False)
            self.train_data_size = train_dataset.size()
            train_dataset = InfiniteDataFlow(train_dataset)
            train_dataset = LocallyShuffleData(train_dataset, 100)
        else:
            train_paths_a = glob(f'./datasets/{self.config.dataset.name}/trainA/*')
            train_paths_b = glob(f'./datasets/{self.config.dataset.name}/trainB/*')
            self.train_data_size = min(len(train_paths_a), len(train_paths_b))
            train_dataset = paths_to_generator(train_paths_a, train_paths_b, True)
            train_dataset = GeneratorToDataFlow(train_dataset, self.train_data_size)

        image_size = (self.config.dataset.image_size, self.config.dataset.image_size)
        train_dataset = ProcessingDataFlow(train_dataset,
                                           crop_size=image_size,
                                           fit_resize=config.dataset.data_loader.fit_resize,
                                           random_fit=config.dataset.data_loader.random_fit,
                                           random_fit_max_ratio=config.dataset.data_loader.random_fit_max_ratio,
                                           random_flip=config.dataset.data_loader.random_flip,
                                           random_brightness=config.dataset.data_loader.random_brightness,
                                           random_contrast=config.dataset.data_loader.random_contrast)
        train_dataset = PrefetchDataZMQ(train_dataset, nr_proc=config.dataset.data_loader.num_proc)
        train_dataset = BatchData(train_dataset, self.config.trainer.batch_size)
        train_dataset.reset_state()
        self.train_dataflow = train_dataset

        # test set
        if config.dataset.data_loader.use_lmdb:
            test_dataset = LMDBDataPoint(f'./datasets/{self.config.dataset.name}/test.lmdb', shuffle=False)
            self.test_data_size = test_dataset.size()
            test_dataset = InfiniteDataFlow(test_dataset)
        else:
            test_paths_a = glob(f'./datasets/{self.config.dataset.name}/testA/*')
            test_paths_b = glob(f'./datasets/{self.config.dataset.name}/testB/*')
            self.test_data_size = min(len(test_paths_a), len(test_paths_b))
            test_dataset = paths_to_generator(test_paths_a, test_paths_b, False)
            test_dataset = GeneratorToDataFlow(test_dataset, self.test_data_size)

        test_dataset = ProcessingDataFlow(test_dataset,
                                          crop_size=image_size,
                                          fit_resize=True,
                                          random_fit=False,
                                          random_flip=False,
                                          random_brightness=False,
                                          random_contrast=False)
        test_dataset = PrefetchDataZMQ(test_dataset, nr_proc=1)
        test_dataset = BatchData(test_dataset, self.config.trainer.batch_size)
        test_dataset.reset_state()
        self.test_dataflow = test_dataset

    def get_train_data_generator(self):
        return self.train_dataflow.get_data()

    def get_validation_data_generator(self):
        raise NotImplementedError

    def get_test_data_generator(self):
        return self.test_dataflow.get_data()

    def get_train_data_size(self):
        return self.train_data_size

    def get_validation_data_size(self):
        raise NotImplementedError

    def get_test_data_size(self):
        return self.test_data_size
