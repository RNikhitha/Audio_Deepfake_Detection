import librosa
import librosa.display
import numpy as np
import os
import pandas as pd

DATASET_PATH = 'release_in_the_wild'

# extracting all .wav files to list
wav_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".wav")]

features_list = []

for file in wav_files:
    file_path = os.path.join(DATASET_PATH, file)

    # audio file
    y, sr = librosa.load(file_path, sr=None)

    # features
    cqtspec = librosa.cqt(y, sr=sr)
    logspec = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    # Mean Pooling
    cqtspec_mean = np.mean(cqtspec, axis=1) 
    logspec_mean = np.mean(logspec, axis=1)
    melspec_mean = np.mean(melspec, axis=1)

    # extracted features are stored in dictionary
    features_list.append({
        "file_name": file,
        "cqtspec": cqtspec_mean.tolist(),
        "logspec": logspec_mean.tolist(),
        "melspec": melspec_mean.tolist()
    })


df = pd.DataFrame(features_list)
df.to_csv("extracted_audio_features_test.csv", index=False)