
import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
from pydub import AudioSegment

# Load SVM model
model = joblib.load('svm_model.pkl')  # Replace 'svm_model.pkl' with your actual model file path

# Define function to extract features from audio files
def extract_features(audio_path):
    # Load MP3 audio file using pydub
    audio = AudioSegment.from_mp3(audio_path)
    # Convert audio to mono
    audio = audio.set_channels(1)
    # Export audio to WAV format (temporary)
    temp_wav = 'temp.wav'
    audio.export(temp_wav, format="wav")
    
    # Extract MFCC features from WAV file
    y, sr = librosa.load(temp_wav, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Remove temporary WAV file
    os.remove(temp_wav)
    
    return mfccs.flatten()


# Get the current working directory
current_dir = os.getcwd()

# Construct the path dynamically
directory = os.path.join(current_dir, 'gtzan.keras-master', 'data', 'samples')

# Directory containing audio files
#directory = 'C:/sp/gtzan.keras-master/data/samples'  # Replace with your actual directory path

# Iterate through audio files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.mp3'):  # Assuming the audio files are in .mp3 format
        audio_path = os.path.join(directory, filename)
        
        # Extract features from the audio file
        features = extract_features(audio_path)
        
        # Print information about the extracted features
        print("Number of features extracted:", len(features))
        print("Shape of features:", features.shape)
        
        # Make prediction using the SVM model
        predicted_label = model.predict([features])[0]
        
        # You might want to map the predicted label to a human-readable class name if needed
        # Example: class_names = ['class1', 'class2', ...]
        # predicted_class_name = class_names[predicted_label]
        
        print(f"Predicted label for {filename}: {predicted_label}")
