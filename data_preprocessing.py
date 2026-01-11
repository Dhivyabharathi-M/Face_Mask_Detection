import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Path to the dataset
data_path = "C:\\Users\\Divya\\OneDrive\\Desktop\\dhivya\\Face mask detection\\data"  # Adjust this if needed

# Paths for masked and non-masked images
mask = os.path.join(data_path, "with_mask")
no_mask = os.path.join(data_path, "without_mask")

# Check the number of images
print("Number of images with mask:", len(os.listdir(mask)))
print("Number of images without mask:", len(os.listdir(no_mask)))

# Prepare images and labels
images, labels = [], []

for category, label in [(mask, 1), (no_mask, 0)]:
    for file in os.listdir(category):
        img_path = os.path.join(category, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize images to 128x128
            images.append(img)
            labels.append(label)

# Convert to numpy arrays and normalize
images = np.array(images) / 255.0
labels = np.array(labels)

# Check for missing values
print("\nChecking for missing values:")
print(pd.DataFrame(images.reshape(len(images), -1)).isnull().sum().sum())

# Visualizing the class distribution
plt.figure(figsize=(6, 4))
plt.bar(['With Mask', 'Without Mask'], [np.sum(labels == 1), np.sum(labels == 0)], color=['green', 'red'])
plt.title('Class Distribution')
plt.xlabel('Category')
plt.ylabel('Number of Images')
plt.show()

# Display a few images from both classes
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i])
    plt.title("Mask" if labels[i] == 1 else "No Mask")
    plt.axis("off")

for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(images[-i - 1])
    plt.title("Mask" if labels[-i - 1] == 1 else "No Mask")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Save the preprocessed data as .npy files for easy loading
np.save("./models/X_train.npy", X_train)
np.save("./models/X_test.npy", X_test)
np.save("./models/y_train.npy", y_train)
np.save("./models/y_test.npy", y_test)

print("\nData preprocessing completed and files saved!")
