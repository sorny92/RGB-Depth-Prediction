import os
import cv2
import numpy as np
from loader import create_train_dataset

dataset = create_train_dataset(os.getenv("DATASET_PATH"), 1)
it = dataset.as_numpy_iterator()
for image, depth in it:
    BGR_image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
    cv2.imshow("image", BGR_image)
    print("Max depth in the image:", np.max(depth[0]))
    cv2.imshow("depth", depth[0]/np.max(depth[0]))
    cv2.waitKey(0)