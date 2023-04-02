import math
import os
from abc import ABC, abstractmethod
from enum import Enum
import pathlib
import random
import tensorflow as tf
import numpy as np


class PHASE(Enum):
    TRAIN = "train"
    TEST = "test"


class DepthLoader(ABC):
    def __init__(self, data_path: pathlib.Path, train_val_split):
        self.data_path = pathlib.Path(data_path)
        self.train_val_split = train_val_split
        self.train_items = []
        self.val_items = []

    @staticmethod
    def __data_generator(items, idx=None):
        if idx is None:
            image_path = random.sample(items, 1)[0]
        else:
            image_path = items[idx]

        yield str(image_path), \
            f'{image_path.with_suffix("")}_depth.npy', \
            f'{image_path.with_suffix("")}_depth_mask.npy'

    def train_data_generator(self):
        def gen():
            for i in range(len(self.train_items)):
                yield next(self.__data_generator(self.train_items))

        return gen

    def val_data_generator(self):
        def gen():
            for i in range(len(self.val_items)):
                data_point = next(self.__data_generator(self.val_items, i))
                yield data_point

        return gen

    @abstractmethod
    def get_pair(self, idx):
        pass


class DIODEDataset(DepthLoader):
    def __init__(self, data_path: pathlib.Path, train_val_split):
        super().__init__(data_path, train_val_split)
        self.images_type = ["indoors", "outdoor"]
        scans = []
        for t in self.images_type:
            for scan in (self.data_path / t).glob("*/*"):
                scans.append(scan)

        n_scans_for_train_set = math.ceil(len(scans) * self.train_val_split)
        random.shuffle(scans)
        for scan in scans[:n_scans_for_train_set]:
            self.train_items.extend(scan.glob("*.png"))
        for scan in scans[n_scans_for_train_set:]:
            self.val_items.extend(scan.glob("*.png"))

    def get_pair(self, idx=None):
        if idx is None:
            random.randint()

    @staticmethod
    @tf.function
    def tf_load_pair(image_path, depth_path, mask_path):
        def load_numpy(numpy_path):
            return np.load(numpy_path.numpy().decode("utf-8"))

        def crop_and_resize(image_to_crop_resize):
            cropped = tf.image.resize_with_crop_or_pad(image_to_crop_resize, 768, 768)
            return tf.image.resize(cropped, (512, 512))

        image = tf.image.decode_png(tf.io.read_file(image_path))
        depth = tf.py_function(load_numpy, inp=[depth_path], Tout=tf.float32)
        mask = tf.expand_dims(tf.py_function(load_numpy, inp=[mask_path], Tout=tf.float32), -1)

        return tf.image.convert_image_dtype(crop_and_resize(image), dtype=tf.uint8), \
            crop_and_resize(depth * mask)


def augment(aug_fn, image, depth):
    aug_image, aug_depth = tf.numpy_function(func=aug_fn, inp=[image, depth],
                                             Tout=[tf.uint8, tf.float32])
    return aug_image, aug_depth


def scale(image, depth):
    return (tf.cast(image, tf.float32) / 255.0), depth


def create_train_val_dataset(dataset_path, aug_fn, train_val_split=0.9):
    dataset = DIODEDataset(dataset_path, train_val_split)
    train_dataset = tf.data.Dataset.from_generator(dataset.train_data_generator(),
                                                   output_types=(tf.string, tf.string, tf.string),
                                                   output_shapes=((), (), ())) \
        .map(DIODEDataset.tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(lambda i, d: augment(aug_fn, i, d),
             num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE) \
        .map(scale, num_parallel_calls=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(dataset.val_data_generator(),
                                                 output_types=(tf.string, tf.string, tf.string),
                                                 output_shapes=((), (), ())) \
        .map(DIODEDataset.tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(scale, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


if __name__ == "__main__":
    path = pathlib.Path(os.getenv("DATASET_PATH")) / "val"
    data = DIODEDataset(path)
    image_path, depth_path, mask_path = next(data.train_data_generator()())
    image, depth = DIODEDataset.tf_load_pair(image_path, depth_path, mask_path)


    def aug(x, y):
        return x, y


    augment(lambda x, y: aug(x, y), image, depth)
    assert np.array(image).dtype == np.uint8
    assert np.array(depth).dtype == np.float32
    assert np.array(image).shape[0] == 512
    assert np.array(depth).shape[0] == 512
    assert np.array(image).shape[:2] == depth.shape[:2]
