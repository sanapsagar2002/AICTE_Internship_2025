# AICTE_Internship_2025_Week_3

# Forest Fire Detection Using Deep Learning

## Overview
This project focuses on detecting forest fires using a Deep Learning model trained on the **Wildfire Dataset** from Kaggle. The model is implemented using TensorFlow and Keras and runs in **Google Colab**.

## Dataset
The dataset is sourced from Kaggle:
- **Name**: [The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)
- **Categories**: Images classified as fire and non-fire
- **Usage**: Training, validation, and testing of the deep learning model

## Features
- Uses **Convolutional Neural Networks (CNNs)** for fire detection
- **Data Augmentation** to improve generalization
- **Evaluation Metrics**: Accuracy, loss, and visualizations
- **Visualization**: Displays dataset samples
- **Colab Compatible**: Designed to run on Google Colab with GPU support
- **Layers**: Includes a custom-built CNN with Conv2D, MaxPooling2D, Dense, and Dropout layers

## Setup and Installation
### Prerequisites
Ensure you have a Google Colab environment and install necessary dependencies:

```python
!pip install kagglehub
```

### Download Dataset
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("elmadafri/the-wildfire-dataset")
print("Path to dataset files:", path)
```

## Project Structure
```
├── forest_fire_detection.ipynb  # Google Colab
├── README.md                     # Project documentation
├── dataset/                       # Contains dataset images
│   ├── train/
│   ├── val/
│   ├── test/
```

## Implementation
### Import Libraries
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
```

### Load and Explore Dataset
```python
train_dir = '/root/.cache/kagglehub/datasets/elmadafri/the-wildfire-dataset/versions/3/the_wildfire_dataset_2n_version/train'
val_dir = '/root/.cache/kagglehub/datasets/elmadafri/the-wildfire-dataset/versions/3/the_wildfire_dataset_2n_version/val'
test_dir = '/root/.cache/kagglehub/datasets/elmadafri/the-wildfire-dataset/versions/3/the_wildfire_dataset_2n_version/test'

# List all classes
classes = os.listdir(train_dir)
num_classes = len(classes)
print(f'Number of Classes: {num_classes}')
print(f'Classes: {classes}')
```

### Visualizing Dataset
```python
plt.figure(figsize=(12, 10))
for i in range(5):
    class_path = os.path.join(train_dir, classes[0])
    img_name = os.listdir(class_path)[i]
    img_path = os.path.join(class_path, img_name)
    img = plt.imread(img_path)
    
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f'{classes[0]} \n shape: {img.shape}')
    plt.axis('off')
plt.show()
```

## Model Training
The project includes a **CNN model** with layers like **Conv2D, MaxPooling, Flatten, Dense, and Dropout** for effective feature extraction and classification.

### Preprocessing
```python
img_width, img_height = 150, 150
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)
```

### Class Mapping
```python
class_mapping = train_generator.class_indices
class_names = list(class_mapping.keys())
print("Class Names:", class_names)
```

### CNN Model Architecture
```python
model = Sequential([
    Input(shape=(img_width, img_height, 3)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

## Training & Evaluation

### Compile & Train
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 32,
    epochs = 12,
    validation_data = val_generator,
    validation_steps = val_generator.samples // 32
)
```

### Plot Accuracy & Loss
```python
# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

### Evaluate on Test Data
```python
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // 32)
print(f'Test Accuracy: {test_acc:.4f}')
```

### Save & Load Model
```python
# Save model
model.save('FFD.keras')

# Load model
from tensorflow.keras.models import load_model
model = load_model('FFD.keras')
```

## Predict Forest Fire on New Image
```python
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def predict_fire(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[1] if prediction[0] > 0.5 else class_names[0]

    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()
```

# Example usage:
```python
predict_fire('/kaggle/input/the-wildfire-dataset/the_wildfire_dataset_2n_version/test/nofire/sample.jpg')
```

## Results & Evaluation
The model is evaluated using accuracy and loss plots, confusion matrix, and test predictions.

## Future Improvements
- Experiment with different architectures (ResNet, VGG16, EfficientNet)
- Use Transfer Learning with pre-trained models like ResNet, VGG16, or EfficientNet
- Add a Streamlit/Flask web app for user interaction
- Expand dataset diversity for better generalization
- Integrate real-time fire detection using drone footage or live feeds

## License
This project is licensed under the MIT License.
