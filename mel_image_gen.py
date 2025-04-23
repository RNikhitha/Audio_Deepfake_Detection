import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

aud_folder = 'release_in_the_wild/'
aud_files = [f for f in os.listdir(aud_folder) if f.endswith('.wav')]

for audio in aud_files:
    audio_file_path = os.path.join(aud_folder, audio)

    # generating melspectograms
    y, sr = librosa.load(audio_file_path, sr=None) 
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    # Plot
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(melspec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram: {audio}')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency (Hz)')

    image_path = audio_file_path.replace('.wav', '_melspec.png')
    plt.savefig(image_path)
    plt.close() 
    print(f"Mel spectrogram for {audio} saved as {image_path}")

