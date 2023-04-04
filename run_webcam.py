import tensorflow as tf
import cv2
import argparse
import numpy as np
from data.loader import scale
import time


class RTProcess:
    def __init__(self, saved_model_path, webcam_device):
        self.model = tf.saved_model.load(saved_model_path)
        self.video_feed = cv2.VideoCapture(webcam_device)

    def process(self):
        start = time.time()
        ret, frame = self.video_feed.read()

        img = cv2.resize(frame, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        img, _ = scale(img, "")

        output = self.model.serve(img).numpy()
        end = time.time()

        output /= np.max(output)
        cv2.putText(frame, f"FPS: {1 / (end - start):0.2f}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255))
        cv2.imshow("Video feed", frame)
        cv2.imshow("Depth map", output[0])
        cv2.putText
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Run with a webcam feed or a video',
        description='Export a tf checkpoint to savedModel and/or quantize the model')
    parser.add_argument('saved_model_path')
    parser.add_argument('input_video')
    args = parser.parse_args()

    process = RTProcess(args.saved_model_path, args.input_video)
    while True:
        process.process()
