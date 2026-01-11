**Face Mask Detection using Convolutional Neural Networks (CNN)**
📌 **Project Overview**

This project implements a Face Mask Detection system using Deep Learning (CNN) to classify whether a person is wearing a face mask or not.
The model is trained from scratch (no pretrained models) using a labeled image dataset and evaluated using standard machine learning metrics.

The system can be extended to real-time webcam detection and is suitable for applications in public safety, healthcare, and surveillance systems.


**🧠 Objectives**

Build a CNN model from scratch for face mask classification

Perform data preprocessing and augmentation

Evaluate the model using Accuracy, Precision, Recall, F1-Score

Visualize performance using confusion matrix and training curves

Ensure reproducibility and clarity for academic submission

**📂 Dataset**

Dataset contains two classes:

with_mask

without_mask

Images are RGB and resized to 128×128

Dataset stored locally in the following structure:

data/
├── with_mask/
└── without_mask/


📌 Dataset Source:
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

**⚙️ Technologies Used**

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Seaborn

Scikit-learn

**🧪 Preprocessing Techniques Used**
Image resizing (128×128)

RGB image validation

Pixel normalization (0–1)

Label encoding (Mask = 1, No Mask = 0)

Train-test split (80:20 with stratification)

Data augmentation:

Rotation

Zoom

Width & height shift

Shear

Horizontal flip

**🏗️ Model Architecture**

Conv2D + ReLU

MaxPooling

Flatten

Dense (Fully Connected Layer)

Dropout (0.5)

Sigmoid Output Layer

Loss Function: Binary Crossentropy
Optimizer: Adam

**📊 Model Evaluation**

The model is evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Training vs Validation Accuracy

Training vs Validation Loss

**📈 Sample Results**

Training Accuracy: ~95–98%

Validation Accuracy: ~90–95%

Clear separation observed in confusion matrix

Stable convergence without severe overfitting

**🗂️ Project Structure**
Face-Mask-Detection/
│
├── data/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── mask_detector_model.h5
│   └── history.pkl
│
├── data_preprocessing.py
├── train_model.py
├── README.md

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install tensorflow keras numpy matplotlib seaborn scikit-learn opencv-python

2️⃣ Preprocess the Dataset
python data_preprocessing.py

3️⃣ Train the Model
python train_model.py


**🙌 Acknowledgements**

Kaggle Dataset Contributors

TensorFlow & Keras Documentation

OpenCV Community
