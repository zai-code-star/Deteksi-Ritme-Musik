# IMPORTING LIBRARIES
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
import tensorflow as tf
from tensorflow import keras
import librosa.display as lplt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import keras as k
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("data/features_3_sec.csv")
df.head()

# Data preprocessing
df.shape
df.dtypes
df = df.drop(labels="filename", axis=1)

audio_recording = "data/genres_original/classical/classical.00000.wav"
data, sr = librosa.load(audio_recording)
print(type(data), type(sr))
librosa.load(audio_recording, sr=45600)

stft = librosa.stft(data)
stft_mag = np.abs(stft)
plt.figure(figsize=(14, 6))
librosa.display.specshow(
    librosa.amplitude_to_db(stft_mag), sr=sr, x_axis="time", y_axis="hz"
)
plt.colorbar()


spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=sr, alpha=0.4, color="#2B4F72")

chroma = librosa.feature.chroma_stft(y=data, sr=sr)
plt.figure(figsize=(16, 6))
lplt.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", cmap="coolwarm")
plt.colorbar()
plt.title("chroma Features")
plt.show()

start = 1000
end = 1200
plt.figure(figsize=(14, 5))
plt.plot(data[start:end], color="#2B4F72")
plt.grid()

zero_cross_rate = librosa.zero_crossings(data[start:end], pad=False)
print("The number of zero-crossings is:", sum(zero_cross_rate))

class_list = df.iloc[:, -1]
convertor = LabelEncoder()

y = convertor.fit_transform(class_list)
y
print(df.iloc[:, :-1])

fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype=float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
len(y_train)
len(y_test)

def trainModel(model, epochs, optimizer):
    batch_size = 128
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
    )


def plotValidate(history):
    print("Validation Accuracy", max(history.history["val_accuracy"]))
    pd.Dataframe(history.history).plot(figsize=(12, 6))
    plt.show()

model = k.models.Sequential(
    [
        k.layers.Dense(512, activation="relu", input_shape=(X_train.shape[1],)),
        k.layers.Dropout(0.2),
        k.layers.Dense(256, activation="relu"),
        k.layers.Dropout(0.2),
        k.layers.Dense(128, activation="relu"),
        k.layers.Dropout(0.2),
        k.layers.Dense(64, activation="relu"),
        k.layers.Dropout(0.2),
        k.layers.Dense(32, activation="softmax"),
    ]
)
print(model.summary())
model_history = trainModel(model=model, epochs=500, optimizer="adam")

# Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate additional metrics
acc_score = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average="weighted")
recall = recall_score(y_test, y_pred_classes, average="weighted")
f1 = f1_score(y_test, y_pred_classes, average="weighted")

print(f"Accuracy: {acc_score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save the model
model.save("musicmd.keras")
