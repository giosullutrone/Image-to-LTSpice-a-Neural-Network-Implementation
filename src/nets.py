def identity_block(x, block_number, sub_block_number, number_of_filters=(32, 32, 128), kernel_size=(3, 3), L2=0.0):
    from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add
    from tensorflow.keras.initializers import he_normal
    from tensorflow.keras.regularizers import l2

    # Retrieve Filters
    f1, f2, f3 = number_of_filters

    # Save the input value
    x_shortcut = x

    # Main path
    # First component of main path
    x = BatchNormalization(axis=3, name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_batch_norm_0")(x)
    x = LeakyReLU()(x)
    x = Conv2D(f1, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_normal(42), kernel_regularizer=l2(L2),
               name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_conv_0")(x)

    # Second component of main path
    x = BatchNormalization(axis=3, name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_batch_norm_1")(x)
    x = LeakyReLU()(x)
    x = Conv2D(f2, kernel_size, strides=(1, 1), padding='same', kernel_initializer=he_normal(42), kernel_regularizer=l2(L2),
               name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_conv_1")(x)

    # Third component of main path
    x = BatchNormalization(axis=3, name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_batch_norm_2")(x)
    x = LeakyReLU()(x)
    x = Conv2D(f3, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_normal(42), kernel_regularizer=l2(L2),
               name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_conv_2")(x)

    # Add
    x = Add()([x, x_shortcut])

    return x


def conv_block(x, block_number, sub_block_number, number_of_filters=(32, 32, 128), L2=0.0):
    from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add
    from tensorflow.keras.initializers import he_normal
    from tensorflow.keras.regularizers import l2

    # Retrieve Filters
    f1, f2, f3 = number_of_filters

    # Save the input value
    x_shortcut = x

    # Main path
    # First component of main path
    x = BatchNormalization(axis=3, name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_batch_norm_0")(x)
    x = LeakyReLU()(x)
    x = Conv2D(f1, (1, 1), strides=(2, 2), padding='valid', kernel_initializer=he_normal(42), kernel_regularizer=l2(L2),
               name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_conv_0")(x)

    # Second component of main path
    x = BatchNormalization(axis=3, name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_batch_norm_1")(x)
    x = LeakyReLU()(x)
    x = Conv2D(f2, (3, 3), strides=(1, 1), padding='same', kernel_initializer=he_normal(42), kernel_regularizer=l2(L2),
               name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_conv_1")(x)

    # Third component of main path
    x = BatchNormalization(axis=3, name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_batch_norm_2")(x)
    x = LeakyReLU()(x)
    x = Conv2D(f3, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_normal(42), kernel_regularizer=l2(L2),
               name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_conv_2")(x)

    # Shortcut
    x_shortcut = BatchNormalization(axis=3, name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_batch_norm_3")(x_shortcut)
    x_shortcut = Conv2D(f3, (1, 1), strides=(2, 2), padding='valid', kernel_initializer=he_normal(42), kernel_regularizer=l2(L2),
                        name="block_" + str(block_number) + "_sub_" + str(sub_block_number) + "_conv_3")(x_shortcut)

    # Add
    x = Add()([x, x_shortcut])

    return x

def __generate_body(input_shape=(416, 416, 1), L2=0.0):
    from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Input, MaxPooling2D
    from tensorflow.keras.initializers import he_normal
    from tensorflow.keras.regularizers import l2

    # Inputs
    inp = Input(input_shape)

    # Pre-processing block ==> Big filter with stride==2 for size reduction
    features = Conv2D(64, kernel_size=(7, 7), padding="same", strides=(2, 2), kernel_initializer=he_normal(42),
                      kernel_regularizer=l2(L2))(inp)
    features = BatchNormalization(axis=3)(features)
    features = LeakyReLU()(features)
    features = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(features)

    # First conv block
    features = conv_block(features, 0, 0, number_of_filters=(64, 64, 256), L2=L2)
    features = identity_block(features, 0, 1, number_of_filters=(64, 64, 256), L2=L2)
    features = identity_block(features, 0, 2, number_of_filters=(64, 64, 256), L2=L2)

    # Second conv block
    features = conv_block(features, 1, 0, number_of_filters=(128, 128, 512), L2=L2)
    features = identity_block(features, 1, 1, number_of_filters=(128, 128, 512), L2=L2)
    features = identity_block(features, 1, 2, number_of_filters=(128, 128, 512), L2=L2)
    features = identity_block(features, 1, 3, number_of_filters=(128, 128, 512), L2=L2)

    # Third conv block
    features = conv_block(features, 2, 0, number_of_filters=(256, 256, 1024), L2=L2)
    features = identity_block(features, 2, 1, number_of_filters=(256, 256, 1024), L2=L2)
    features = BatchNormalization(axis=3)(features)
    features = LeakyReLU()(features)

    return inp, features

def generate_pre_tracking_model(input_shape=(416, 416, 1), number_of_classes=7, L2=0.0):
    from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation
    from tensorflow.keras.initializers import glorot_normal
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model

    inp, features = __generate_body(input_shape, L2)

    output = Conv2D(number_of_classes, kernel_size=(1, 1), padding="same", strides=(1, 1), kernel_initializer=glorot_normal(42),
                    kernel_regularizer=l2(L2), activation='sigmoid', name="out_base")(features)

    output = Flatten()(output)

    output = Dense(number_of_classes, kernel_initializer=glorot_normal(42), name="out_pre_tracking")(output)
    output = Activation("softmax")(output)

    model = Model(
        inputs=inp,
        outputs=output,
    )

    return model

def generate_tracking_model(input_shape=(416, 416, 1), output_shape=(13, 13, 11), number_of_classes=6, L2=0.0):
    from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Concatenate
    from tensorflow.keras.initializers import glorot_normal
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model

    inp, features = __generate_body(input_shape, L2)

    output = Conv2D(number_of_classes + 1, kernel_size=(1, 1), padding="same", strides=(1, 1), kernel_initializer=glorot_normal(42),
                    kernel_regularizer=l2(L2), activation='sigmoid', name="out_base")(features)

    output = Conv2D(256, kernel_size=(1, 1), padding="same", strides=(1, 1),
                    kernel_initializer=glorot_normal(42), kernel_regularizer=l2(L2),
                    activation=None, name="FinalConv0")(output)
    output = BatchNormalization()(output)
    output = Activation("sigmoid")(output)

    output = Conv2D(256, kernel_size=(3, 3), padding="same", strides=(1, 1),
                    kernel_initializer=glorot_normal(42), kernel_regularizer=l2(L2),
                    activation=None, name="FinalConv1")(output)
    output = BatchNormalization()(output)
    output = Activation("sigmoid")(output)

    output = Conv2D(256, kernel_size=(1, 1), padding="same", strides=(1, 1),
                    kernel_initializer=glorot_normal(42), kernel_regularizer=l2(L2),
                    activation=None, name="FinalConv2")(output)
    output = BatchNormalization()(output)
    output = Activation("sigmoid")(output)

    output = Conv2D(256, kernel_size=(3, 3), padding="same", strides=(1, 1),
                    kernel_initializer=glorot_normal(42), kernel_regularizer=l2(L2),
                    activation=None, name="FinalConv3")(output)
    output = BatchNormalization()(output)
    output = Activation("sigmoid")(output)

    output = Conv2D(256, kernel_size=(1, 1), padding="same", strides=(1, 1),
                    kernel_initializer=glorot_normal(42), kernel_regularizer=l2(L2),
                    activation=None, name="FinalConv4")(output)
    output = BatchNormalization()(output)
    output = Activation("sigmoid")(output)

    output_data = Conv2D(output_shape[-1] - number_of_classes, kernel_size=(3, 3), padding="same", strides=(1, 1),
                         kernel_initializer=glorot_normal(42), kernel_regularizer=l2(L2), activation='sigmoid', name="out_data")(output)

    output_type = Conv2D(number_of_classes, kernel_size=(3, 3), padding="same", strides=(1, 1),
                         kernel_initializer=glorot_normal(42), kernel_regularizer=l2(L2),
                         activation='softmax', name="out_type")(output)

    output = Concatenate()([output_data, output_type])

    model = Model(
        inputs=inp,
        outputs=output,
    )

    return model

def generate_identification_model(number_of_classes, input_shape=(50, 50, 1), L2=0.0):
    from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D
    from tensorflow.keras.initializers import glorot_normal
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model

    inp, features = __generate_body(input_shape, L2)

    output = GlobalAveragePooling2D()(features)

    output = Dense(number_of_classes, kernel_initializer=glorot_normal(42), kernel_regularizer=l2(L2), name="out_id")(output)
    output = Activation("softmax")(output)

    model = Model(
        inputs=inp,
        outputs=output,
    )

    return model
