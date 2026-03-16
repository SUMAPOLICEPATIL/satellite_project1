import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def create_model(num_classes):

    # Load pretrained ResNet50
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Summary:\n")
    model.summary()

    return model


def train_model(model, train_data, val_data):

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )

    # Save trained model
    model.save("satellite_model.h5")

    print("\nModel saved as satellite_model.h5")

    return history