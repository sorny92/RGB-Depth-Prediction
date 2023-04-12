import os
import pathlib
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from model import UNET, DeeplabV3Plus
from data.loader import create_train_val_dataset
from data.data_augmentation import aug_fn


def SSIMLoss(y_true, y_pred):
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))) / 2


def SobelLoss(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    return tf.reduce_mean(tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred))


def custom_loss(y_true, y_pred):
    alpha = 0.3
    return alpha * keras.losses.mean_absolute_error(y_true, y_pred) \
        + SobelLoss(y_true, y_pred) + SSIMLoss(y_true, y_pred)


def scaled_mae(y_true, y_pred):
    depth_mean = 3.600834
    depth_std = 4.563519
    # tf.math.log(depth + 1.0) / tf.math.log(tf.constant([20.0]))
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.math.exp(y_pred * tf.math.log(tf.constant([20.0]))) - 1.0

    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.math.exp(y_true * tf.math.log(tf.constant([20.0]))) - 1.0
    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)

    model = DeeplabV3Plus(image_size=512)
    EPOCHS = 1000

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss=custom_loss,
        metrics=[scaled_mae],
    )

    dataset_path = os.getenv("DATASET_PATH")

    BATCH_SIZE = 8
    train_data_path = pathlib.Path(dataset_path) / "train"
    train_dataset, val_dataset = create_train_val_dataset(train_data_path, aug_fn, BATCH_SIZE, 0.9)

    from datetime import datetime

    run_id = f'{model.name}-{datetime.now().strftime("%m%d-%H%M%S")}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"models/{run_id}/" + "{epoch:02d}-{val_scaled_mae:.2f}",
        save_weights_only=True,
        monitor='val_scaled_mae',
        mode='max',
        save_best_only=False)

    log_dir = f"./logs/{run_id}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100,
                                                          profile_batch=5)
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm', max_queue=2)


    def log_validation_images(batch, logs):
        if batch % 100 != 0:
            return

        def image_grid(input, prediction):
            img_mean = tf.constant([98.42558877, 91.09861066, 86.32245039])
            img_std = tf.constant([44.93833011, 45.5801206, 49.50943415])
            depth_mean = 3.600834
            depth_std = 4.563519
            import numpy as np
            # Create a figure to contain the plot.
            figure = plt.figure(figsize=(20, 10))
            for idx, i in enumerate([input[0][0], input[1][0], prediction]):
                # Start next subplot.
                plt.subplot(1, 3, idx + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(i)
            plt.tight_layout()
            figure.savefig(f"images/{run_id}_{datetime.now().strftime('%m%d-%H%M%S')}.png",
                           format='png')
            plt.close(figure)

        # Use the model to predict the values from the validation dataset.
        in_image = val_dataset.take(1)
        test_pred_raw = model.predict(in_image)[0]
        input_image_as_np = in_image.as_numpy_iterator().next()

        image_grid(input_image_as_np, test_pred_raw)


    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                              patience=20)
    # Define the per-epoch callback.
    images_callback = keras.callbacks.LambdaCallback(on_batch_end=log_validation_images)

    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=EPOCHS,
                        callbacks=[model_checkpoint_callback,
                                   tensorboard_callback,
                                   images_callback,
                                   reduce_lr_callback])
