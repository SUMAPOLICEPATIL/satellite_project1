import tensorflow as tf

IMG_SIZE = (224,224)
BATCH_SIZE = 32

def load_data():

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_data.class_names

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    return train_data, val_data, class_names, data_augmentation