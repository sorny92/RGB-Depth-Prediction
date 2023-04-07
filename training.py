import os
import pathlib
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from model import UNET
from data.loader import create_train_val_dataset
from data.data_augmentation import aug_fn


def SSIMLoss(y_true, y_pred):
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))) / 2


def SobelLoss(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    return tf.reduce_mean(tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred))


def custom_loss(y_true, y_pred):
    alpha = 0.1
    return alpha * keras.losses.mean_absolute_error(y_true, y_pred) \
        + SobelLoss(y_true, y_pred) + SSIMLoss(y_true, y_pred)


def scaled_mae(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    # tf.log(2.71*(depth/40)+1))
    y_pred = (tf.math.exp(y_pred) - 1) * 40 / 2.71
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = (tf.math.exp(y_true) - 1) * 40 / 2.71
    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)

    model = UNET(image_size=512)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.0001, amsgrad=True),
        loss=custom_loss,
        metrics=[scaled_mae],
    )

    dataset_path = os.getenv("DATASET_PATH")

    BATCH_SIZE = 4
    train_data_path = pathlib.Path(dataset_path) / "train"
    train_dataset, val_dataset = create_train_val_dataset(train_data_path, aug_fn, BATCH_SIZE)

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
        if not batch % 100:
            print("LOL")
            return
        print("LOL2")

        def plot_to_image(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            figure.savefig(buf, format='png')
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image

        def image_grid(input, prediction):
            import numpy as np
            # Create a figure to contain the plot.
            figure = plt.figure(figsize=(20, 10))
            for idx, i in enumerate([input[0][0], input[1][0], prediction]):
                if i.dtype == np.float32:
                    i = np.copy(i)
                    i /= np.max(i)
                    i *= 255.0
                    i = i.astype(np.uint8)

                # Start next subplot.
                plt.subplot(1, 3, idx + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(i)
                plt.tight_layout()
            return figure

        # Use the model to predict the values from the validation dataset.
        in_image = val_dataset.take(1)
        test_pred_raw = model.predict(in_image)[0]
        input_image_as_np = in_image.as_numpy_iterator().next()

        figure = image_grid(input_image_as_np, test_pred_raw)
        cm_image = plot_to_image(figure)

        with file_writer_cm.as_default():
            res = tf.summary.image("epoch_validation_images", cm_image, step=batch)


    # Define the per-epoch callback.
    images_callback = keras.callbacks.LambdaCallback(on_batch_end=log_validation_images)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                              patience=10)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_scaled_mae",
        mode="max",
        patience=7,
        restore_best_weights=True)

    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=100,
                        callbacks=[model_checkpoint_callback, tensorboard_callback,
                                   reduce_lr_callback])
