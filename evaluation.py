from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_confusion_matrix(model, val_data, class_names):

    y_true = []
    y_pred = []

    for images, labels in val_data:

        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)

        y_true.extend(labels)
        y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.show()