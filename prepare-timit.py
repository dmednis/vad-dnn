import numpy as np
import librosa
from glob import glob
from os import path

files = glob('raw/voice/*.WAV')


for file in files:
    data, fs = librosa.load(file)

    fft_size = 1024
    overlap_fac = 0.5

    data = data / np.max(data)

    # hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    hop_size = np.int32(fs / 100)
    pad_end_size = fft_size  # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
    t_max = len(data) / np.float32(fs)

    window = np.hanning(fft_size)  # our half cosine window
    inner_pad = np.zeros(fft_size)  # the zeros which will be used to double each segment size

    proc = np.concatenate((data, np.zeros(pad_end_size)))  # the data to process
    result = np.empty((total_segments, fft_size), dtype=np.float32)  # space to hold the result

    for i in range(total_segments):  # for each segment
        current_hop = hop_size * i  # figure out the current segment offset
        segment = proc[current_hop:current_hop + fft_size]  # get the current segment
        windowed = segment * window  # multiply by the half cosine function

        padded = np.append(windowed, inner_pad)  # add 0s to double the length of the data

        spectrum = np.fft.fft(padded) / fft_size  # take the Fourier Transform and scale by the number of samples

        autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum

        result[i, :] = autopower[:fft_size]  # append to the results array

    result = 20 * np.log10(result)

    np.save('data/' + path.basename(file), result)
