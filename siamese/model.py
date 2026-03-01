import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, BatchNormalization, Dropout

from siamese.config import IMG_SIZE, IMG_CHANNELS, EMBEDDING_DIM


def make_embedding() -> Model:
    inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS), name="input_image")

    x = Conv2D(64, (10, 10), activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D(64, (2, 2), padding="same")(x)

    x = Conv2D(128, (7, 7), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(64, (2, 2), padding="same")(x)

    x = Conv2D(128, (4, 4), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(64, (2, 2), padding="same")(x)

    x = Conv2D(256, (4, 4), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(EMBEDDING_DIM, activation="sigmoid")(x)

    return Model(inputs=[inp], outputs=[x], name="embedding")


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

    def get_config(self):
        return super().get_config()


def make_siamese_model(embedding: Model) -> Model:
    input_image = Input(name="input_img", shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS))
    validation_image = Input(name="validation_img", shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS))

    distance_layer = L1Dist()
    distance_layer._name = "distance"
    distances = distance_layer(embedding(input_image), embedding(validation_image))

    x = Dense(512, activation="relu")(distances)
    x = Dropout(0.3)(x)
    classifier = Dense(1, activation="sigmoid")(x)

    return Model(
        inputs=[input_image, validation_image],
        outputs=classifier,
        name="SiameseNetwork",
    )
