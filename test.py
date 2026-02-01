import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import pickle
import librosa  
import librosa.display
from IPython.display import Audio  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import io
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


model = load_model("musicmd.keras")

def predict(audio_file):
    data, sr = librosa.load(audio_file)

    stft = librosa.stft(data)
    stft_mag = np.abs(stft)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
    chroma = librosa.feature.chroma_stft(y=data, sr=sr)
    zero_cross_rate = librosa.zero_crossings(data, pad=False)

    min_length = min(len(stft_mag), len(spectral_rolloff), len(chroma.T), len(zero_cross_rate))
    stft_mag = stft_mag[:, :min_length]
    spectral_rolloff = spectral_rolloff[:min_length]
    chroma = chroma.T[:min_length]
    zero_cross_rate = zero_cross_rate[:min_length]

    fitx = StandardScaler()
    input_features = np.hstack((stft_mag.T, spectral_rolloff.reshape(-1, 1), chroma, zero_cross_rate.reshape(-1, 1)))
    input_features = fitx.fit_transform(input_features[:, :58])

    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    # Make predictions
    convertor = LabelEncoder()
    fity = StandardScaler()
    convertor.fit(genres)

    predictions = model.predict(input_features)
    predicted_genre_index = np.argmax(predictions[0])
    predicted_genre = convertor.inverse_transform([predicted_genre_index])[0]
    
    return predicted_genre