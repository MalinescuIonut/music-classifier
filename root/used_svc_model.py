import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# Get the current working directory
current_dir = os.getcwd()

# Construct the path dynamically
gtzan_dir = os.path.join(current_dir, 'gtzan.keras-master', 'data', 'genres')


# Path to the GTZAN dataset directory
#gtzan_dir = 'C:\\sp\\gtzan.keras-master\\data\\genres'

# Function to extract audio features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Concatenate features
    features = np.concatenate((zero_crossing_rate,
                               spectral_centroid,
                               spectral_contrast,
                               spectral_bandwidth,
                               spectral_rolloff,
                               mfcc,
                               chroma), axis=0)
    
    # Return mean values of each feature
    return np.mean(features, axis=1)

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

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Define hyperparameters for grid search
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'rbf']
}

# Perform grid search with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print("Best Model Hyperparameters:", grid_search.best_params_)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

"""
Training Accuracy: 0.99875
Testing Accuracy: 0.695



Fitting 5 folds for each of 8 candidates, totalling 40 fits
[CV] END .....................svm__C=0.1, svm__kernel=linear; total time=   0.0s
[CV] END .....................svm__C=0.1, svm__kernel=linear; total time=   0.0s
[CV] END .....................svm__C=0.1, svm__kernel=linear; total time=   0.0s
[CV] END .....................svm__C=0.1, svm__kernel=linear; total time=   0.0s
[CV] END .....................svm__C=0.1, svm__kernel=linear; total time=   0.0s
[CV] END ........................svm__C=0.1, svm__kernel=rbf; total time=   0.0s
[CV] END ........................svm__C=0.1, svm__kernel=rbf; total time=   0.0s
[CV] END ........................svm__C=0.1, svm__kernel=rbf; total time=   0.0s
[CV] END ........................svm__C=0.1, svm__kernel=rbf; total time=   0.0s
[CV] END ........................svm__C=0.1, svm__kernel=rbf; total time=   0.0s
[CV] END .......................svm__C=1, svm__kernel=linear; total time=   0.0s
[CV] END .......................svm__C=1, svm__kernel=linear; total time=   0.0s
[CV] END .......................svm__C=1, svm__kernel=linear; total time=   0.0s
[CV] END .......................svm__C=1, svm__kernel=linear; total time=   0.0s
[CV] END .......................svm__C=1, svm__kernel=linear; total time=   0.0s
[CV] END ..........................svm__C=1, svm__kernel=rbf; total time=   0.0s
[CV] END ..........................svm__C=1, svm__kernel=rbf; total time=   0.0s
[CV] END ..........................svm__C=1, svm__kernel=rbf; total time=   0.0s
[CV] END ..........................svm__C=1, svm__kernel=rbf; total time=   0.0s
[CV] END ..........................svm__C=1, svm__kernel=rbf; total time=   0.0s
[CV] END ......................svm__C=10, svm__kernel=linear; total time=   0.0s
[CV] END ......................svm__C=10, svm__kernel=linear; total time=   0.0s
[CV] END ......................svm__C=10, svm__kernel=linear; total time=   0.0s
[CV] END ......................svm__C=10, svm__kernel=linear; total time=   0.0s
[CV] END ......................svm__C=10, svm__kernel=linear; total time=   0.0s
[CV] END .........................svm__C=10, svm__kernel=rbf; total time=   0.0s
[CV] END .........................svm__C=10, svm__kernel=rbf; total time=   0.0s
[CV] END .........................svm__C=10, svm__kernel=rbf; total time=   0.0s
[CV] END .........................svm__C=10, svm__kernel=rbf; total time=   0.0s
[CV] END .........................svm__C=10, svm__kernel=rbf; total time=   0.0s
[CV] END .....................svm__C=100, svm__kernel=linear; total time=   0.7s
[CV] END .....................svm__C=100, svm__kernel=linear; total time=   0.2s
[CV] END .....................svm__C=100, svm__kernel=linear; total time=   0.1s
[CV] END .....................svm__C=100, svm__kernel=linear; total time=   0.3s
[CV] END .....................svm__C=100, svm__kernel=linear; total time=   0.2s
[CV] END ........................svm__C=100, svm__kernel=rbf; total time=   0.0s
[CV] END ........................svm__C=100, svm__kernel=rbf; total time=   0.0s
[CV] END ........................svm__C=100, svm__kernel=rbf; total time=   0.0s
[CV] END ........................svm__C=100, svm__kernel=rbf; total time=   0.0s
[CV] END ........................svm__C=100, svm__kernel=rbf; total time=   0.0s
Best Model Hyperparameters: {'svm__C': 100, 'svm__kernel': 'rbf'}
Training Accuracy: 0.99875
Testing Accuracy: 0.695
Model saved successfully to svm_model.pkl
"""

