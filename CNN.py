# import libraries
import numpy as np
import os
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import keras

@keras.saving.register_keras_serializable()
class SpecAugment(tf.keras.layers.Layer):
    def __init__(self, freq_mask=8, time_mask=15, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def call(self, inputs, training=None):
        if not training:
            return inputs

        x = inputs

        # Frequency masking
        freq_width = tf.random.uniform(
            shape=(), minval=0, maxval=self.freq_mask + 1, dtype=tf.int32
        )
        freq_start = tf.random.uniform(
            shape=(), minval=0, maxval=64 - freq_width + 1, dtype=tf.int32
        )
        freq_mask_tensor = tf.concat(
            [
                tf.ones((freq_start, 130, 1)),
                tf.zeros((freq_width, 130, 1)),
                tf.ones((64 - freq_start - freq_width, 130, 1)),
            ],
            axis=0,
        )
        x = x * freq_mask_tensor

        # Time masking
        time_width = tf.random.uniform(
            shape=(), minval=0, maxval=self.time_mask + 1, dtype=tf.int32
        )
        time_start = tf.random.uniform(
            shape=(), minval=0, maxval=130 - time_width + 1, dtype=tf.int32
        )
        time_mask_tensor = tf.concat(
            [
                tf.ones((64, time_start, 1)),
                tf.zeros((64, time_width, 1)),
                tf.ones((64, 130 - time_start - time_width, 1)),
            ],
            axis=1,
        )
        x = x * time_mask_tensor

        return x

    # Required for proper model loading in other scripts
    def get_config(self):
        config = super().get_config()
        config.update({
            "freq_mask": self.freq_mask,
            "time_mask": self.time_mask,
        })
        return config

if __name__ == '__main__':
    ser_folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/"

    """ CNN"""

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    def add_data(input_dir, output_arr, label_arr):
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".pt"):
                file_path = os.path.join(input_dir, file_name)
                data = torch.load(file_path)

                # Convert PyTorch tensor → NumPy
                data = data.numpy()

                # (1, 64, time) → (64, time)
                data = data.squeeze(0)

                # Force time dimension to exactly 130
                if data.shape[1] < 130:
                    # Pad with zeros
                    pad_width = 130 - data.shape[1]
                    data = np.pad(
                        data,
                        ((0, 0), (0, pad_width)),
                        mode='constant'
                    )

                elif data.shape[1] > 130:
                    # Truncate
                    data = data[:, :130]

                # (64, 130) → (64, 130, 1)
                data = np.expand_dims(data, axis=-1)

                # Append the training and testing data to the respective lists
                output_arr.append(data)

                # Extract the label from the file name
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

    add_data(ser_folder+"/train/angry/", x_train, y_train)
    add_data(ser_folder+"/train/disgust/", x_train, y_train)
    add_data(ser_folder+"/train/fear/", x_train, y_train)
    add_data(ser_folder+"/train/happy/", x_train, y_train)
    add_data(ser_folder+"/train/neutral/", x_train, y_train)
    add_data(ser_folder+"/train/sad/", x_train, y_train)
    add_data(ser_folder+"/train/surprise/", x_train, y_train)

    add_data(ser_folder+"/test/angry/", x_test, y_test)
    add_data(ser_folder+"/test/disgust/", x_test, y_test)
    add_data(ser_folder+"/test/fear/", x_test, y_test)
    add_data(ser_folder+"/test/happy/", x_test, y_test)
    add_data(ser_folder+"/test/neutral/", x_test, y_test)
    add_data(ser_folder+"/test/sad/", x_test, y_test)
    add_data(ser_folder+"/test/surprise/", x_test, y_test)

    # convert lists to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # convert labels to one-hot
    y_train_oh = tf.keras.utils.to_categorical(y_train, 7)
    y_test_oh = tf.keras.utils.to_categorical(y_test, 7)

    print(y_train_oh)
    print(y_test_oh)

    # print for debugging
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    print(torch.load(ser_folder+"/test/angry/angry1.pt").shape)

    test_acc = []
    best_model = None
    best_acc = 0
    best_num = 1

    # repeat 5 times
    for i in range(1):
        # create CNN structure
        model = Sequential()
        model.add(Input(shape=(64, 130, 1)))
        model.add(SpecAugment(freq_mask=5, time_mask=10))

        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (2, 2), activation='relu'))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(64, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        model.summary()
    test_acc = []
    best_model = None
    best_acc = 0
    best_num = 1

    # repeat 5 times
    for i in range(1):
        # create CNN structure
        model = Sequential()
        model.add(Input(shape=(64, 130, 1)))
        model.add(SpecAugment(freq_mask=5, time_mask=10))

        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (2, 2), activation='relu'))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(64, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        model.summary()

        # compile and train model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(x_train, y_train_oh, epochs=100, batch_size=32)
        # compile and train model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(x_train, y_train_oh, epochs=100, batch_size=32)

        # evaluate accuracy
        y_pred = model.predict(x_test)
        y_pred = tf.argmax(y_pred, axis=1)
        # y_test = tf.argmax(y_test_oh, axis=1) 
        loss, acc = model.evaluate(x_test, y_test_oh)

        best_acc = max(best_acc, acc)
        best_model = model if acc == best_acc else best_model
        best_num = i+1 if acc == best_acc else best_num

        test_acc.append(acc)
        print("Test accuracy:", acc)
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    print("Average test accuracy:", np.mean(test_acc))
    print("Best test accuracy:", best_acc)
    print("Best model:", best_num)
    print("Confusion matrix for best model:\n", confusion_matrix(y_test, tf.argmax(best_model.predict(x_test), axis=1)))

    best_model.save("/Users/cici/Documents/VS Code/CodingBara/best_model.keras")
