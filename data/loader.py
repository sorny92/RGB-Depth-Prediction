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
    def __init__(self, data_path: pathlib.Path, train_val_split: float = 0.8):
        self.data_path = pathlib.Path(data_path)
        self.train_val_split = train_val_split
        self.train_items = []
        self.val_items = []
        self.current_val_idx = 0

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
            yield next(self.__data_generator(self.train_items))

        return gen

    def val_data_generator(self):
        def gen():
            data_point = next(self.__data_generator(self.val_items, self.current_val_idx))
            self.current_val_idx += 1
            yield data_point

        return gen

    @abstractmethod
    def get_pair(self, idx):
        pass


class DIODEDataset(DepthLoader):
    def __init__(self, data_path: pathlib.Path, train_val_split: float = 0.8):
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
        image = tf.image.decode_png(tf.io.read_file(image_path))

        def load_numpy(path):
            return np.load(path.numpy().decode("utf-8"))

        depth = tf.py_function(load_numpy, inp=[depth_path], Tout=tf.float32)
        mask = tf.py_function(load_numpy, inp=[mask_path], Tout=tf.float32)
        mask = tf.expand_dims(mask, -1)

        return image, depth * mask


def create_train_dataset(dataset_path, batch_size):
    data = DIODEDataset(dataset_path)
    dataset_generator = tf.data.Dataset.from_generator(data.train_data_generator(),
                                                       output_types=(tf.string, tf.string, tf.string),
                                                       output_shapes=((), (), ())).repeat()

    return dataset_generator.map(DIODEDataset.tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size).repeat()


if __name__ == "__main__":
    data = DIODEDataset(os.getenv("DATASET_PATH"))
    image, depth = next(data.val_data_generator()())
    image, depth = DIODEDataset.tf_load_pair(image, depth)
    assert np.array(image).shape[0:2] == depth.shape[:2]
