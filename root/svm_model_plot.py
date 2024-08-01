import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import random

# Function to extract audio features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0, 0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0, 0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0, 0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0, 0]
    
    # Return all four features
    return np.array([zero_crossing_rate, spectral_centroid, spectral_bandwidth, spectral_rolloff])

# Load pre-trained SVM model
model_path = 'svm_model.pkl'
svm_model = joblib.load(model_path)

# Get the current working directory
current_dir = os.getcwd()

# Construct the path dynamically
sample_dir = os.path.join(current_dir, 'gtzan.keras-master', 'data', 'samples')

# Collect paths to all MP3 audio samples
audio_paths = []
for root, dirs, files in os.walk(sample_dir):
    for file in files:
        if file.endswith('.mp3'):
            audio_paths.append(os.path.join(root, file))

# Shuffle and select 10% of the data
random.shuffle(audio_paths)
selected_audio_paths = audio_paths[:max(1, int(0.1 * len(audio_paths)))]

# Extract features from audio samples
features_list = []
for audio_path in selected_audio_paths:
    features = extract_features(audio_path)
    features_list.append(features)
audio_features = np.array(features_list)

# Scale the features
scaler = StandardScaler()
audio_features_scaled = scaler.fit_transform(audio_features)

# Predict labels for audio samples
predicted_labels = svm_model.predict(audio_features_scaled)

# Plot the audio samples on a 2D plot using the first two features
plt.figure(figsize=(8, 6))
for i, label in enumerate(predicted_labels):
    if label == 0:
        color = 'blue'
        marker = 'o'
    elif label == 1:
        color = 'red'
        marker = 's'
    plt.scatter(audio_features_scaled[i, 0], audio_features_scaled[i, 1], color=color, marker=marker, label=f'Sample {i+1}')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.title('Visualization of Audio Samples (First 2 Features)')
plt.legend()
plt.grid(True)
plt.show()
