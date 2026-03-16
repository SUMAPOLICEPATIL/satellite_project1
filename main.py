from data_preprocessing import load_data
from model_training import create_model, train_model
from visualization import plot_results
from prediction import predict_image
from evaluation import show_confusion_matrix
from gradcam import gradcam


def main():

    print("\nLoading Dataset...\n")

    train_data, val_data, class_names, data_aug = load_data()

    print("\nClasses detected:\n")
    print(class_names)

    print("\nCreating Deep Learning Model...\n")

    model = create_model(len(class_names))

    print("\nTraining Model...\n")

    history = train_model(model, train_data, val_data)

    print("\nPlotting Training Results...\n")

    plot_results(history)

    print("\nGenerating Confusion Matrix...\n")

    show_confusion_matrix(model, val_data, class_names)

    print("\nPredicting Test Image...\n")

    predict_image("test.jpg", class_names)

    print("\nGenerating Grad-CAM Heatmap...\n")

    gradcam(model, "test.jpg")


if __name__ == "__main__":
    main()