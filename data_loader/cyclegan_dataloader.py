import os
from datetime import datetime
from glob import glob
from typing import Optional, List, Generator, Tuple

import numpy as np
from dotmap import DotMap
from tensorpack.dataflow import PrefetchDataZMQ, BatchData, DataFlow

from base.base_data_loader import BaseDataLoader
from utils.dataflow import GeneratorToDataFlow
from utils.image import read_image, resize_image_by_ratio, normalize_image


class ProcessingDataFlow(DataFlow):
    def __init__(self, ds: DataFlow, crop_size: Optional[Tuple[int, int]], fit_resize: bool, random_flip: bool,
                 random_brightness: bool, random_contrast: bool,
                 random_fit: bool, random_fit_max_ratio: float = 1.3) -> None:
        self.ds = ds
        self.crop_size = crop_size
        self.fit_resize = fit_resize
        self.random_fit_max_ratio = random_fit_max_ratio
        self.random_fit = random_fit
        self.random_flip = random_flip
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast

    def reset_state(self) -> None:
        super().reset_state()

        # set random seed per process
        seed = (id(self) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
        np.random.seed(seed)

        self.ds.reset_state()

    def __iter__(self):
        for path_a, path_b in self.ds.__iter__():
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
            yield normalize_image(img_a), normalize_image(img_b)

    def __len__(self):
        return self.ds.__len__()


def paths_to_generator(paths_a: List[str], paths_b: List[str], shuffle: bool) -> Generator:
    while True:
        if shuffle:
            np.random.shuffle(paths_a)
            np.random.shuffle(paths_b)

        for path_a, path_b in zip(paths_a, paths_b):
            yield path_a, path_b


class CycleGanDataLoader(BaseDataLoader):
    def __init__(self, config: DotMap) -> None:
        super().__init__(config)

        train_paths_a = glob(f'./datasets/{self.config.dataset.name}/trainA/*')
        train_paths_b = glob(f'./datasets/{self.config.dataset.name}/trainB/*')
        self.train_data_size: int = min(len(train_paths_a), len(train_paths_b))
        image_size = (self.config.dataset.image_size, self.config.dataset.image_size)

        train_dataset = paths_to_generator(train_paths_a, train_paths_b, True)
        train_dataset = GeneratorToDataFlow(train_dataset)
        train_dataset = ProcessingDataFlow(ds=train_dataset,
                                           crop_size=image_size,
                                           fit_resize=config.dataset.data_loader.fit_resize,
                                           random_fit_max_ratio=config.dataset.data_loader.random_fit_max_ratio,
                                           random_flip=config.dataset.data_loader.random_flip,
                                           random_brightness=config.dataset.data_loader.random_brightness,
                                           random_contrast=config.dataset.data_loader.random_contrast,
                                           random_fit=config.dataset.data_loader.random_fit)
        train_dataset = PrefetchDataZMQ(train_dataset, num_proc=config.dataset.data_loader.num_proc)
        train_dataset = BatchData(train_dataset, self.config.trainer.batch_size)
        train_dataset.reset_state()
        self.train_dataflow: DataFlow = train_dataset

        # valid set
        valtestset_a = sorted(glob(f'./datasets/{self.config.dataset.name}/testA/*'))
        valtestset_b = sorted(glob(f'./datasets/{self.config.dataset.name}/testB/*'))
        valtestset_size = min(len(valtestset_a), len(valtestset_b))
        valid_paths_a = valtestset_a[:valtestset_size // 2]
        valid_paths_b = valtestset_b[:valtestset_size // 2]
        self.valid_data_size: int = min(len(valid_paths_a), len(valid_paths_b))

        valid_dataset = paths_to_generator(valid_paths_a, valid_paths_b, False)
        valid_dataset = GeneratorToDataFlow(valid_dataset)
        valid_dataset = ProcessingDataFlow(ds=valid_dataset,
                                           crop_size=image_size,
                                           fit_resize=True,
                                           random_flip=False,
                                           random_brightness=False,
                                           random_contrast=False,
                                           random_fit=False, )
        valid_dataset = PrefetchDataZMQ(valid_dataset, num_proc=1)
        valid_dataset = BatchData(valid_dataset, self.config.trainer.batch_size)
        valid_dataset.reset_state()
        self.valid_dataflow: DataFlow = valid_dataset

        # test set
        test_paths_a = valtestset_a[valtestset_size // 2:]
        test_paths_b = valtestset_b[valtestset_size // 2:]
        self.test_data_size: int = min(len(test_paths_a), len(test_paths_b))

        test_dataset = paths_to_generator(test_paths_a, test_paths_b, False)
        test_dataset = GeneratorToDataFlow(test_dataset)
        test_dataset = ProcessingDataFlow(ds=test_dataset,
                                          crop_size=image_size,
                                          fit_resize=True,
                                          random_flip=False,
                                          random_brightness=False,
                                          random_contrast=False,
                                          random_fit=False, )
        test_dataset = PrefetchDataZMQ(test_dataset, num_proc=1)
        test_dataset = BatchData(test_dataset, self.config.trainer.batch_size)
        test_dataset.reset_state()
        self.test_dataflow: DataFlow = test_dataset

    def get_train_data_generator(self) -> Generator:
        return self.train_dataflow.get_data()

    def get_validation_data_generator(self) -> Generator:
        return self.valid_dataflow.get_data()

    def get_test_data_generator(self) -> Generator:
        return self.test_dataflow.get_data()

    def get_train_data_size(self) -> int:
        return self.train_data_size

    def get_validation_data_size(self) -> int:
        return self.valid_data_size

    def get_test_data_size(self) -> int:
        return self.test_data_size
