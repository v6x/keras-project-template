import cv2
import numpy as np


def read_image(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)


def normalize_image(x):
    return x.astype(np.float32) / 127.5 - 1 if x is not None else None


def denormalize_image(x):
    return np.array(x * 127.5 + 127.5, dtype=np.uint8) if x is not None else None


def denormalize_tensor(tensor):
    return (tensor + 1.) * 127.5


def resize_image(image, output_size):
    return cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_LINEAR)


def resize_image_by_ratio(image, ratio):
    return cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
