import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def gradcam(model, image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img,(224,224))

    img_array = np.expand_dims(img/255.0, axis=0)

    last_conv_layer = model.get_layer("conv5_block3_out")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    plt.matshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.show()