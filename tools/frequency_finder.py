import numpy as np
from scipy.signal import find_peaks

def main_frequency(x, fs):
    
    x = np.asarray(x)
    x = x - np.mean(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1/fs)

    mag = np.abs(X)
    mag[0] = 0
    id_max = np.argmax(mag)
    
    peaks, _ = find_peaks(mag, height=0.5*mag[id_max])

    freqs_peaks = freqs[peaks]

    differences = np.diff(freqs_peaks)

    avg_diff = np.median(differences)

    return avg_diff

def top_frequencies(x, fs, n=10):
    x = np.asarray(x)

    # remove DC offset
    x = x - np.mean(x)

    # optional window
    #window = np.hanning(len(x))
    #xw = x * window

    # FFT
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1/fs)

    # magnitude spectrum
    mag = np.abs(X)

    # ignore DC
    mag[0] = 0

    # find spectral peaks
    peaks, _ = find_peaks(mag)

    # sort peaks by amplitude
    strongest = peaks[np.argsort(mag[peaks])[::-1]]

    # take top n
    strongest = strongest[:n]

    return list(zip(freqs[strongest], mag[strongest]))