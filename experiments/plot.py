import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram


def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


def plot_waves(_sound_names, _raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(_sound_names, _raw_sounds):
        plt.subplot(10, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_specgram(_sound_names, _raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(_sound_names, _raw_sounds):
        plt.subplot(10, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram', x=0.5, y=0.915, fontsize=18)
    plt.show()


sound_file_paths = ["raw/sample2.wav"]
sound_names = ["sample2"]

raw_sounds = load_sound_files(sound_file_paths)

plot_waves(sound_names, raw_sounds)

