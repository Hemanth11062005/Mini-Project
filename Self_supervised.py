# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout, Input, Lambda, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
import tensorflow.keras.backend as K

# %%
# Load and Display Image
img_path = 'Dataset/pest/train/aphids/jpg_0 - Copy (2).jpg'
if os.path.exists(img_path):
    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
else:
    print(f"Warning: Image path {img_path} not found. Skipping display.")


# %%
train_dir = 'Dataset/pest/train'
test_dir = 'Dataset/pest/test'

# %%
# Image Data Generator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    zca_epsilon=1e-06,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.2,
    shear_range=20,
    zoom_range=0.8,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.05,
    rescale=1./255  # Add rescaling to normalize pixel values
)


# %%
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# %%
# Load data
training = train_datagen.flow_from_directory(
    train_dir, 
    batch_size=32, 
    target_size=(224, 224), 
    subset="training"
)


# %%
validing = train_datagen.flow_from_directory(
    train_dir, 
    batch_size=32, 
    target_size=(224, 224), 
    subset='validation', 
    shuffle=True
)

# %%
testing = test_datagen.flow_from_directory(
    test_dir, 
    batch_size=32, 
    target_size=(224, 224), 
    shuffle=True
)

# %%
# Get number of classes
num_classes = len(training.class_indices)
print(f"Number of classes: {num_classes}")
print(f"Class mapping: {training.class_indices}")

# %%
# Build a model with MobileNetV2 (more compatible across TF versions)
def build_model():
    # Use MobileNetV2 as base model
    base_model = MobileNetV2(weights='imagenet', 
                            include_top=False, 
                            input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# %%
# Train and evaluate function
def train_and_evaluate():
    # Dictionary to store results
    results = {}
    models = {}
    
    # Build and train the model
    print("Training model...")
    model = build_model()
    history = model.fit(
        training,
        validation_data=validing,
        epochs=20,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ],
        verbose=2
    )
    
    # Evaluate on test data
    test_loss, test_acc = model.evaluate(testing)
    
    # Store results
    model_name = "MobileNetV2_Transfer"
    results[model_name] = test_acc
    models[model_name] = model
    
    # Save model
    model.save(f"{model_name}.h5")
    print(f"Model Test Accuracy: {test_acc * 100:.2f}%")
    
    return results, models


# %%
# Train and get the models
try:
    print("Starting training...")
    results, models = train_and_evaluate()
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"Best model is {best_model_name} with accuracy {results[best_model_name] * 100:.2f}%")
    
    # Test on a single image
    img_test_path = 'Dataset/pest/test/beetle/jpg_33.jpg'
    if os.path.exists(img_test_path):
        # Load and preprocess the image
        img_test = image.load_img(img_test_path, target_size=(224, 224))
        plt.imshow(img_test)
        plt.title("Test Image")
        plt.show()
        
        img_array = image.img_to_array(img_test)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Get class labels
        class_labels = list(training.class_indices.keys())
        
        # Make predictions
        prediction = best_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        
        print(f"\nBest Model ({best_model_name}) Predicted Class: {class_labels[predicted_class[0]]}")
        
        # Display prediction probabilities
        plt.figure(figsize=(10, 5))
        plt.bar(class_labels, prediction[0])
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.title(f"Model Prediction Probabilities")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_prediction.png')
        plt.show()
    else:
        print(f"Warning: Test image path {img_test_path} not found. Skipping prediction.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Here's the full traceback:")
    import traceback
    traceback.print_exc()