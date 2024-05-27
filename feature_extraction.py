import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pickle

def load_model_and_preprocessors(model_path, pca_path, scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, pca, scaler

def process_and_extract_features(df):
    fs_original = 1000  # Original sampling frequency
    fs_new = 200  # New target sampling frequency
    nyquist_new = fs_new / 2

    # Design a low-pass filter for downsampling (to avoid aliasing)
    b_downsample, a_downsample = signal.butter(5, nyquist_new / (fs_original / 2), btype='low')

    # Design bandpass filter for the target frequency range (1 Hz to 75 Hz)
    lowcut = 1.0  # Low cut-off frequency
    highcut = 75.0  # High cut-off frequency
    low = lowcut / nyquist_new
    high = highcut / nyquist_new
    b_bandpass, a_bandpass = signal.butter(5, [low, high], btype='band')

    # Initialize container for downsampled and filtered data
    processed_data_df = pd.DataFrame()

    # Minimum length required for filtfilt to avoid ValueError
    min_length = 33

    # Process each channel
    for column in df.columns:
        data = df[column].values  # Extract data for each channel

        if len(data) < min_length:
            raise ValueError(f"Data in column '{column}' is too short for filtering")

        # Apply the low-pass filter for downsampling
        filtered_data_for_downsampling = signal.filtfilt(b_downsample, a_downsample, data)

        # Decimate (downsample) - take every 5th sample
        downsampled_data = filtered_data_for_downsampling[::5]

        # Apply the bandpass filter
        final_filtered_data = signal.filtfilt(b_bandpass, a_bandpass, downsampled_data)
        
        # Store in new dataframe
        processed_data_df[column] = final_filtered_data

    # Extract features
    window_length_sec = 4  # Window length is 4 seconds
    window_length_samples = window_length_sec * fs_new  # Convert window length from seconds to samples

    # Define frequency bands
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 14), 'Beta': (14, 31), 'Gamma': (31, 50)}
    num_bands = len(bands)
    num_channels = len(df.columns)
    total_samples = processed_data_df.shape[0]
    num_windows = total_samples // window_length_samples  # Number of complete 4-second windows

    # Initialize arrays for PSD and DE features
    psd_features = np.zeros((num_bands, num_channels, num_windows))
    de_features = np.zeros((num_bands, num_channels, num_windows))

    for i, (band, (low_freq, high_freq)) in enumerate(bands.items()):
        nyquist = fs_new / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(5, [low, high], btype='band')

        for j in range(num_channels):
            column = processed_data_df.columns[j]
            data = processed_data_df[column].values

            if len(data) < min_length:
                raise ValueError(f"Data in column '{column}' is too short for filtering")

            # Filter data for specific band
            filtered_data = signal.filtfilt(b, a, data)

            # Segment and calculate features for each window
            for k in range(num_windows):
                start = k * window_length_samples
                end = start + window_length_samples
                segment = filtered_data[start:end]

                # Calculate PSD using Welch's method
                freqs, power = signal.welch(segment, fs_new, nperseg=window_length_samples)
                psd_features[i, j, k] = np.mean(power)

                # Calculate DE
                de_features[i, j, k] = 0.5 * np.log2(2 * np.pi * np.exp(1) * np.var(segment))

    # Convert numpy arrays to pandas DataFrames
    psd_features_df = [pd.DataFrame(psd_features[i]) for i in range(psd_features.shape[0])]
    de_features_df = [pd.DataFrame(de_features[i]) for i in range(de_features.shape[0])]

    # Find the maximum number of columns (Wmax)
    max_cols = max(df.shape[1] for df in psd_features_df + de_features_df)

    # Pad each DataFrame with zeros to match the max number of columns (Wmax)
    padded_features = [df.join(pd.DataFrame(0, index=df.index, columns=range(df.shape[1], max_cols))) for df in psd_features_df + de_features_df]

    # Flatten each DataFrame
    flattened_features = [df.values.flatten() for df in padded_features]

    return np.array(flattened_features)

def predict_emotion_from_features(features, model, pca, scaler):
    # Normalize each flattened array (MinMax scaling)
    normalized_features = scaler.transform(features)

    # Apply PCA
    pca_features = pca.transform(normalized_features)

    # Predict emotion
    predictions = model.predict(pca_features)
    return predictions

def predict_emotion(df, model, pca, scaler):
    features = process_and_extract_features(df)
    return predict_emotion_from_features(features, model, pca, scaler)
