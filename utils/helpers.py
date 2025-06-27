
import numpy as np



def smooth_signal(signal_vec, window_size=5):
    signal_cumsum = np.cumsum(signal_vec)
    signal_cumsum[window_size:] = (
        signal_cumsum[window_size:] - signal_cumsum[:-window_size]
    )
    smoothed_signal = signal_cumsum[window_size - 1 :] / window_size
    return smoothed_signal



