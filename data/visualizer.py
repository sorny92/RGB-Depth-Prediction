import os
import pathlib
import cv2
import numpy as np
from data.loader import create_train_val_dataset
from data.data_augmentation import aug_fn


def visualize_images(dataset_path):
    dataset, _ = create_train_val_dataset(dataset_path, aug_fn, 1, 0.5)
    it = dataset.repeat().as_numpy_iterator()
    for image, depth in it:
        BGR_image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
        cv2.imshow("image", BGR_image)
        print("Max depth in the image:", np.max(BGR_image))
        print("Max depth in the image:", np.max(depth[0]))
        print("Min depth in the image:", np.min(BGR_image))
        print("Min depth in the image:", np.min(depth[0]))
        cv2.imshow("depth", depth[0] / np.max(depth[0]))
        cv2.waitKey(0)


if __name__ == "__main__":
    visualize_images(pathlib.Path(os.getenv("DATASET_PATH")) / "val")
