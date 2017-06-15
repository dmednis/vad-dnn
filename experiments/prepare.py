import numpy
import librosa
import csv


def speech(time, vad_labels):
    status = 0
    for vad in vad_labels:
        if time > vad['start'] and time < vad['end']:
            status = 1
            break
    return status


def load_raw_labels():
    labels_filename = 'raw/labels4.txt'
    vad_labels = list()
    with open(labels_filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            vad_labels.append({
                'start': float(row[0]),
                'end': float(row[1])
            })
    return vad_labels

raw_labels = load_raw_labels()

audio_filename = 'raw/audio4mono.wav'
data, sr = librosa.load(audio_filename)
audio_len = len(data) / sr

audio = list()
labels = list()


i = 0
while i < len(data):
    start = i
    end = i + 1024
    if end >= len(data):
        end = len(data)
    labels.append(speech(start/sr, raw_labels))
    audio.append(numpy.fft.rfft(data[start:end], 1024))
    i += 1024


numpy.save('data/audio4.npy', audio)
numpy.save('data/vad4.npy', labels)
print('DONE')


