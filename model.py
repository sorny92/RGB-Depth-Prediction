import tensorflow as tf
from tensorflow import keras
from keras import regularizers, initializers, layers


def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_size):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


def UNET(image_size):
    def upsampling(input_tensor, n_filters, concat_layer):
        '''
        Constitutes the block of Decoder
        '''
        # Bilinear 2x upsampling layer
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(input_tensor)
        # concatenation with encoder block
        x = layers.concatenate([x, concat_layer])
        # decreasing the depth filters by half
        x = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        return x

    # Layer name of encoders to be concatenated
    names = ['pool3_pool', 'pool2_pool', 'pool1', 'conv1/relu']
    # Transfer learning approach without the classification head
    encoder = keras.applications.DenseNet169(include_top=False, weights='imagenet',
                                             input_shape=(image_size, image_size, 3))
    # Model build
    inputs = encoder.input
    x = encoder.output
    # decoder blocks linked with corresponding encoder blocks
    bneck = layers.Conv2D(filters=1664, kernel_size=(1, 1), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(bneck)
    x = upsampling(bneck, 832, encoder.get_layer(names[0]).output)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 416, encoder.get_layer(names[1]).output)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 208, encoder.get_layer(names[2]).output)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = upsampling(x, 104, encoder.get_layer(names[3]).output)
    x = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)

    return keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    model = UNET(image_size=512)
    model.summary(line_length=160)
