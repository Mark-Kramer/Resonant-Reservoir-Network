# ------------------------------------------------------------------
# 5) Feature Extraction Functions
# ------------------------------------------------------------------

import numpy as np

def extract_fft_features(sequences, fs=1000, fmin=5, fmax=300):
    """
    For each sequence in `sequences`, compute the FFT and extract
    frequency bins within [fmin..fmax], along with their negative-frequency
    counterparts. Then return real and imaginary parts as features.
    """
    
    def freq_to_fft_bin(freq, N, fs):
        """
        Convert a target frequency (Hz) to the nearest FFT bin index
        for a DFT of length N with sampling rate fs.
        """
        return int(round(freq * N / fs))
    
    feature_list = []
    
    for seq in sequences:
        # seq is shape (T,)
        N = seq.shape[0]
        fft_vals = np.fft.fft(seq, axis=0)  # shape (T,)
        
        # Compute the bin indices for the requested frequency range
        pos_start = freq_to_fft_bin(fmin, N, fs)
        pos_end   = freq_to_fft_bin(fmax, N, fs)
        
        # Make sure these are in valid range [1..(N-1)]
        # (bin 0 is DC; typically skip or keep if needed, but here we skip)
        pos_start = max(1, pos_start)
        pos_end   = min(pos_end, N - 1)
        
        if pos_start > pos_end:
            # If fmin is larger than Nyquist, or something unexpected,
            # skip or produce zero features
            feature_list.append(np.zeros(0))  # or handle differently
            continue
        
        # Negative frequency indices:
        # For an N-point FFT, the negative freq bin k corresponds to index N-k
        # so we want [N - pos_end .. N - pos_start].
        neg_start = N - pos_end
        neg_end   = N - pos_start

        # Build index arrays
        pos_freqs = np.arange(pos_start, pos_end + 1)
        neg_freqs = np.arange(neg_start, neg_end + 1)
        
        # Select from fft_vals
        fft_selected = fft_vals[np.concatenate([pos_freqs, neg_freqs])]
        
        # Real and imaginary parts
        real_part = np.real(fft_selected)
        imag_part = np.imag(fft_selected)
        
        # Concatenate into one feature vector
        features = np.concatenate([real_part, imag_part])
        feature_list.append(features)
    
    # Because not all sequences necessarily yield the same number of freq bins,
    # you might want to ensure they do. Typically you should have consistent T 
    # in each sequence, but handle dimension mismatch if T can vary across examples.
    #
    # If T is the same for all sequences, you’ll get a consistent shape. 
    return np.array(feature_list, dtype=object)

def extract_fft_power_features(sequences, fs=1000, fmin=5, fmax=250):
    """
    For each sequence in `sequences`, compute the FFT and extract
    the power (magnitude squared) of frequency bins within [fmin..fmax],
    including negative-frequency counterparts. Each sequence then yields
    a feature vector of size:
        number_of_selected_bins (positive + negative).
    """
    
    def freq_to_fft_bin(freq, N, fs):
        """
        Convert a target frequency (Hz) to the nearest FFT bin index
        for a DFT of length N with sampling rate fs.
        """
        return int(round(freq * N / fs))

    feature_list = []
    
    for seq in sequences:
        # seq is shape (T,)
        N = seq.shape[0]
        #fft_vals = np.fft.fft(seq, axis=0)  # shape (T,)
        
        # only positive half
        pos_bins = np.fft.rfftfreq(N, 1/fs)
        mask     = (pos_bins >= fmin) & (pos_bins <= fmax)
        power    = np.abs(np.fft.rfft(seq)[mask])**2
        feature_list.append(power)
    
    # Convert to a consistent 2D numeric array
    # (Assumin g all sequences have same length => same # bins.)
    return np.array(feature_list, dtype=object)