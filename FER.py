# import libraries
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

print("GPUs Available:", tf.config.list_physical_devices('GPU'))

fer_folder = "/Users/cici/Documents/VS Code/CodingBara/FER/"

""" Data Loading """

x_train = []
x_test = []
y_train = []
y_test = []

def add_data(input_dir, output_arr, label_arr):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(input_dir, file_name)
            data = Image.open(file_path)
            data = data.convert('L')

            if data.size != (96, 96):
                print(f"!!! DIMENSION MISMATCH ALERT !!!: {file_name}")

            data = np.array(data, dtype="float32")
            data /= 255.0
            output_arr.append(data)

            element = file_name[:2]
            if element == 'an':
                label_arr.append(0)
            elif element == 'di':
                label_arr.append(1)
            elif element == 'ha':
                label_arr.append(2)
            elif element == 'fe':
                label_arr.append(3)
            elif element == 'ne':
                label_arr.append(4)
            elif element == 'sa':
                label_arr.append(5)
            elif element == 'su':
                label_arr.append(6)
            else:
                raise ValueError(f"Unknown emotion in filename: {file_name}")

# Load Train Data
add_data(fer_folder+"train/angry/", x_train, y_train)
add_data(fer_folder+"train/disgust/", x_train, y_train)
add_data(fer_folder+"train/fear/", x_train, y_train)
add_data(fer_folder+"train/happy/", x_train, y_train)
add_data(fer_folder+"train/neutral/", x_train, y_train)
add_data(fer_folder+"train/sad/", x_train, y_train)
add_data(fer_folder+"train/surprise/", x_train, y_train)

# Load Test Data
add_data(fer_folder+"test/angry/", x_test, y_test)
add_data(fer_folder+"test/disgust/", x_test, y_test)
add_data(fer_folder+"test/fear/", x_test, y_test)
add_data(fer_folder+"test/happy/", x_test, y_test)
add_data(fer_folder+"test/neutral/", x_test, y_test)
add_data(fer_folder+"test/sad/", x_test, y_test)
add_data(fer_folder+"test/surprise/", x_test, y_test)

# Convert lists to numpy arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Add channel dimension
if len(x_train.shape) == 3:
    x_train = np.expand_dims(x_train, axis=-1)
if len(x_test.shape) == 3:
    x_test = np.expand_dims(x_test, axis=-1)

# Train/Val Split
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

# Convert labels to one-hot
y_train_oh = tf.keras.utils.to_categorical(y_train, 7)
y_val_oh = tf.keras.utils.to_categorical(y_val, 7)
y_test_oh = tf.keras.utils.to_categorical(y_test, 7)

print(y_train_oh)
print(y_test_oh)

# print for debugging
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Calculate Class Weights to handle dataset imbalance
class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_vals))

test_acc = []
best_model = None
best_acc = 0
best_num = 1

# Remove ImageDataGenerator completely from your script
# Instead, add augmentation layers directly into your model architecture:

for i in range(3):
    model = Sequential([
        Input(shape=(96, 96, 1)),
        
        # Augmentation layers built into the model (active during training only)
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        tf.keras.layers.RandomZoom(0.08),

        # Block 1 (96x96 -> 48x48)
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Block 2 (48x48 -> 24x24)
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 3 (24x24 -> 12x12)
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # Block 4 (12x12 -> 6x6)
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    # Clean, direct fit call without generator overhead
    model.fit(
        x_train, y_train_oh,
        batch_size=64,
        epochs=100,
        validation_data=(x_val, y_val_oh),
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # Evaluate on Test Set
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    loss, acc = model.evaluate(x_test, y_test_oh)
    test_acc.append(acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_num = i + 1

    print("Test accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

print("Average test accuracy:", np.mean(test_acc))
print("Best test accuracy:", best_acc)
print("Best model:", best_num)

# Save the best performing model
best_model.save("/Users/cici/Documents/VS Code/CodingBara/best_FER_model.keras")
