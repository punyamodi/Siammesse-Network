import os
import tensorflow as tf

from siamese.config import (
    ANC_PATH,
    POS_PATH,
    NEG_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    TRAIN_SPLIT,
)


def preprocess(file_path: str) -> tf.Tensor:
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def augment(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return tf.clip_by_value(img, 0.0, 1.0)


def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label


def preprocess_twin_augmented(input_img, validation_img, label):
    anchor = augment(preprocess(input_img))
    positive = augment(preprocess(validation_img))
    return anchor, positive, label


def build_dataset(
    sample_size: int = 300,
    batch_size: int = BATCH_SIZE,
    use_augmentation: bool = False,
):
    anchor = tf.data.Dataset.list_files(os.path.join(ANC_PATH, "*.jpg")).take(sample_size)
    positive = tf.data.Dataset.list_files(os.path.join(POS_PATH, "*.jpg")).take(sample_size)
    negative = tf.data.Dataset.list_files(os.path.join(NEG_PATH, "*.jpg")).take(sample_size)

    positives = tf.data.Dataset.zip(
        (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor))))
    )
    negatives = tf.data.Dataset.zip(
        (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))))
    )
    data = positives.concatenate(negatives)

    map_fn = preprocess_twin_augmented if use_augmentation else preprocess_twin
    data = data.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    train_size = round(len(data) * TRAIN_SPLIT)
    train_data = (
        data.take(train_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_data = (
        data.skip(train_size)
        .take(round(len(data) * (1 - TRAIN_SPLIT)))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_data, test_data
