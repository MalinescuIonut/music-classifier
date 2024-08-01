import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, AveragePooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import ProgbarLogger
import matplotlib.pyplot as plt

# Get the current working directory
current_dir = os.getcwd()

# Construct the path dynamically
gtzan_dir = os.path.join(current_dir, 'gtzan.keras-master', 'data', 'genres')

#gtzan_dir = 'C:\\sp\\gtzan.keras-master\\data\\genres'

# Function to extract audio features
def extract_features(file_path, max_length=1000):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Concatenate features
    features = np.vstack((zero_crossing_rate, spectral_centroid, spectral_contrast, spectral_bandwidth, spectral_rolloff, mfcc, mfcc_delta))
    
    # Pad or truncate features to max_length
    if features.shape[1] < max_length:
        features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_length]
    
    return features

# Load audio files and extract features
def load_data(dataset_path):
    audio_data = []
    labels = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                try:
                    features = extract_features(file_path)
                    audio_data.append(features)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}. Skipping...")
                    continue

    return np.array(audio_data), np.array(labels)

# Load data
audio_data, labels = load_data(gtzan_dir)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_data, encoded_labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Reshape data for CNN input
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Define CNN model
model = Sequential([
    Input(shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], 1)),
    Conv2D(32, (3, 3), padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), padding='same'),
    LeakyReLU(alpha=0.1),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256),
    LeakyReLU(alpha=0.1),
    Dense(128),
    LeakyReLU(alpha=0.1),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define progress bar logger callback
progbar_logger = ProgbarLogger()

# Train the model with progress bar logger callback
history=model.fit(X_train_cnn, y_train_categorical, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test_categorical), callbacks=[progbar_logger])

# Save the model
model.save('classifier_model.h5')
model.save('classifier_model.keras')

# Evaluate the model
loss, accuracy = model.evaluate(X_test_cnn, y_test_categorical)
print("Testing Loss:", loss)
print("Testing Accuracy:", accuracy)


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()


'''
25/25 ━━━━━━━━━━━━━━━━━━━━ 10s 314ms/step - accuracy: 0.0788 - loss: 216.1180 - val_accuracy: 0.1100 - val_loss: 4.1918
Epoch 2/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 292ms/step - accuracy: 0.1550 - loss: 3.0374 - val_accuracy: 0.1800 - val_loss: 2.4277
Epoch 3/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 291ms/step - accuracy: 0.2515 - loss: 2.2084 - val_accuracy: 0.2050 - val_loss: 2.2178
Epoch 4/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 6s 255ms/step - accuracy: 0.2839 - loss: 2.0221 - val_accuracy: 0.2500 - val_loss: 2.2279
Epoch 5/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 261ms/step - accuracy: 0.3678 - loss: 1.9349 - val_accuracy: 0.2850 - val_loss: 2.0144
Epoch 6/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 291ms/step - accuracy: 0.3386 - loss: 1.8190 - val_accuracy: 0.2550 - val_loss: 2.0985
Epoch 7/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 292ms/step - accuracy: 0.3751 - loss: 1.6748 - val_accuracy: 0.2350 - val_loss: 2.2212
Epoch 8/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 8s 312ms/step - accuracy: 0.4311 - loss: 1.6585 - val_accuracy: 0.3550 - val_loss: 1.8402
Epoch 9/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 280ms/step - accuracy: 0.4571 - loss: 1.4773 - val_accuracy: 0.3250 - val_loss: 1.9138
Epoch 10/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 289ms/step - accuracy: 0.4735 - loss: 1.4270 - val_accuracy: 0.3100 - val_loss: 2.0033
Epoch 11/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 287ms/step - accuracy: 0.5182 - loss: 1.4142 - val_accuracy: 0.3500 - val_loss: 1.8391
Epoch 12/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 8s 303ms/step - accuracy: 0.5479 - loss: 1.2528 - val_accuracy: 0.3450 - val_loss: 2.0649
Epoch 13/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 295ms/step - accuracy: 0.5433 - loss: 1.3616 - val_accuracy: 0.3450 - val_loss: 1.8605
Epoch 14/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 295ms/step - accuracy: 0.6236 - loss: 1.1219 - val_accuracy: 0.3650 - val_loss: 2.0639
Epoch 15/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 8s 310ms/step - accuracy: 0.6543 - loss: 1.0218 - val_accuracy: 0.3900 - val_loss: 1.9259
Epoch 16/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 289ms/step - accuracy: 0.6135 - loss: 1.1303 - val_accuracy: 0.3200 - val_loss: 1.9976
Epoch 17/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 286ms/step - accuracy: 0.6993 - loss: 0.9628 - val_accuracy: 0.3850 - val_loss: 1.8208
Epoch 18/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 291ms/step - accuracy: 0.7395 - loss: 0.8003 - val_accuracy: 0.2900 - val_loss: 2.3257
Epoch 19/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 286ms/step - accuracy: 0.6006 - loss: 1.0797 - val_accuracy: 0.3400 - val_loss: 2.0972
Epoch 20/20
25/25 ━━━━━━━━━━━━━━━━━━━━ 7s 288ms/step - accuracy: 0.7172 - loss: 0.8028 - val_accuracy: 0.3300 - val_loss: 2.1785
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 49ms/step - accuracy: 0.3033 - loss: 2.2889
Testing Loss: 2.1784744262695312
Testing Accuracy: 0.33000001311302185


2nd test
Testing Loss: 1.9121382236480713
Testing Accuracy: 0.3700000047683716
'''