# Satellite Image Classification using Deep Learning
## 1. Project Overview
This project builds a **Deep Learning system** that classifies satellite images into different land-use categories such as forest, river, highway, and residential areas.
The model learns patterns from satellite images and predicts the **type of land area** in a new image.
The project uses the **EuroSAT dataset**, which contains satellite images of different land categories.
---
## 2. Objectives
The main objectives of this project are:
* Classify satellite images using Deep Learning
* Use **Transfer Learning (ResNet50)** to improve accuracy
* Visualize model performance using graphs
* Evaluate results using a **confusion matrix**
* Explain model predictions using **Grad-CAM heatmaps**
---
## 3. Dataset
The project uses the **EuroSAT dataset**.
The dataset contains **10 land use classes**:
* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake
Each folder contains satellite images belonging to that category.
Dataset structure:
```
dataset/
 ├── AnnualCrop
 ├── Forest
 ├── HerbaceousVegetation
 ├── Highway
 ├── Industrial
 ├── Pasture
 ├── PermanentCrop
 ├── Residential
 ├── River
 └── SeaLake
```
---
## 4. Project Structure

```
satellite_project
│
├── dataset
│
├── data_preprocessing.py
├── model_training.py
├── prediction.py
├── visualization.py
├── evaluation.py
├── gradcam.py
├── main.py
└── README.md
```
Description of files:

* **data_preprocessing.py**
  Loads the dataset and prepares training and validation data.
* **model_training.py**
  Creates the deep learning model using ResNet50 and trains it.
* **prediction.py**
  Predicts the class of a new satellite image.
* **visualization.py**
  Displays training accuracy and loss graphs.
* **evaluation.py**
  Generates a confusion matrix to evaluate the model.
* **gradcam.py**
  Shows Grad-CAM heatmaps to explain model predictions.
* **main.py**
  Runs the complete pipeline.
---

## 5. Technologies Used
Programming Language:
* Python
Libraries:
* TensorFlow
* NumPy
* Matplotlib
* OpenCV
* Scikit-learn
* Seaborn
* Pandas
---

## 6. Output
The program will:
1. Load the satellite dataset
2. Train the deep learning model
3. Display training accuracy graphs
4. Show a confusion matrix
5. Predict the class of a new image
6. Generate a Grad-CAM heatmap showing where the model focused

---

## 7. Results
The model learns to classify satellite images into different land categories.
Transfer learning with **ResNet50** improves performance compared to a basic CNN model.

---

## 8. Conclusion

This project demonstrates how deep learning can be used to analyze satellite imagery.
Such systems can help in **environment monitoring, land-use analysis, and urban planning**.
