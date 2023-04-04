from model import DeeplabV3Plus
from data.loader import create_train_val_dataset
from data.data_augmentation import aug_fn
from tensorflow import keras
import tensorflow as tf
from training import custom_loss
import argparse


def ThresholdAccuracy(y_true, y_pred, threshold=1.25):
    ranges = tf.maximum(y_true / y_pred, y_pred / y_true)
    threshold = tf.cast(threshold, y_pred.dtype)
    count = tf.cast(ranges < threshold, y_pred.dtype)
    return tf.reduce_mean(count)


def delta1(y_true, y_pred):
    return ThresholdAccuracy(y_true, y_pred)


def delta2(y_true, y_pred):
    return ThresholdAccuracy(y_true, y_pred, threshold=1.25 * 1.25)


def delta3(y_true, y_pred):
    return ThresholdAccuracy(y_true, y_pred, threshold=1.25 * 1.25 * 1.25)


def test(model_checkpoint, model, dataset_path):
    model.load_weights(model_checkpoint)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=custom_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(),
                 tf.keras.metrics.RootMeanSquaredError(),
                 delta1, delta2, delta3],
    )
    _, test_dataset = create_train_val_dataset(dataset_path, aug_fn, 4, train_val_split=0)

    print(model.evaluate(test_dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Evaluate model',
        description='Export a tf checkpoint to savedModel and/or quantize the model')
    parser.add_argument('checkpoint')
    parser.add_argument('dataset')
    args = parser.parse_args()

    test(args.checkpoint, DeeplabV3Plus(512), args.dataset)
