import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Get the current working directory
current_dir = os.getcwd()

# Construct the path dynamically
gtzan_dir = os.path.join(current_dir, 'gtzan.keras-master', 'data', 'genres')

# Function to extract MFCC features
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

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
                    mfcc = extract_mfcc(file_path)
                    audio_data.append(mfcc)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}. Skipping...")
                    continue

    return np.array(audio_data), np.array(labels)

# Load GTZAN dataset
audio_data, labels = load_data(gtzan_dir)

# Data preprocessing
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(audio_data, encoded_labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoded vectors
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# Model architecture
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(13,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_onehot, epochs=100, batch_size=32, validation_data=(X_test, y_test_onehot), verbose=1)

# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train, y_train_onehot, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Save the model
model.save('classifier_model_mfcc.h5')
model.save('classifier_model_mfcc.keras')

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
Training Accuracy: 0.9900000095367432
Testing Accuracy: 0.5649999976158142
'''