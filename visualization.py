import matplotlib.pyplot as plt

def plot_results(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()

    plt.plot(acc,label="Train Accuracy")
    plt.plot(val_acc,label="Validation Accuracy")

    plt.legend()
    plt.title("Model Accuracy")

    plt.show()

    plt.figure()

    plt.plot(loss,label="Train Loss")
    plt.plot(val_loss,label="Validation Loss")

    plt.legend()
    plt.title("Model Loss")

    plt.show()