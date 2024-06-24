import os
import random
import pickle
import pandas as pd
import numpy as np
from scipy.signal import resample, welch
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from tqdm import tqdm

epsilon = 1e-12  # small value to avoid null values

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def power_spectral_density(signal, fs=200):
    freqs, psd = welch(signal, fs)
    return freqs, psd


def spectral_entropy(psd):
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log(psd_norm + epsilon))


def differential_entropy(signal):
    sigma2 = np.var(signal)
    return 0.5 * np.log(2 * np.pi * np.e * sigma2)


def compute_log_cov_features(statistics):
    features = np.array(list(statistics.values()))
    features = features[:144]
    features_matrix = features.reshape((12, 12))
    
    cov_matrix = np.cov(features_matrix)
    log_cov_matrix = np.log(np.abs(cov_matrix) + epsilon)
    
    upper_triangular_indices = np.triu_indices_from(log_cov_matrix)
    log_cov_features = log_cov_matrix[upper_triangular_indices]
    
    return log_cov_features


def extract_statistics(window):
    stats = {}
    
    for col in window.columns:
        signal = window[col].values
        stats[f'mean_{col}'] = np.mean(signal)
        stats[f'std_{col}'] = np.std(signal)
        stats[f'skew_{col}'] = skew(signal)
        stats[f'kurtosis_{col}'] = kurtosis(signal)
        stats[f'max_{col}'] = np.max(signal)
        stats[f'min_{col}'] = np.min(signal)
        
        # Derivatives of 0.5s windows
        half_len = len(signal) // 2
        first_half = signal[:half_len]
        second_half = signal[half_len:]
        stats[f'deriv_max_{col}'] = np.max(second_half) - np.max(first_half)
        stats[f'deriv_min_{col}'] = np.min(second_half) - np.min(first_half)
        
        # Log-Energy Entropy for 0.5-second window
        log_energy_entropy_first_half = np.sum(np.log(first_half**2 + epsilon))
        log_energy_entropy_second_half = np.sum(np.log(second_half**2 + epsilon))
        stats[f'log_energy_entropy_first_half_{col}'] = log_energy_entropy_first_half
        stats[f'log_energy_entropy_second_half_{col}'] = log_energy_entropy_second_half
        
        # Derivatives of 0.25s windows
        quarter_len = len(signal) // 4
        quarters = [signal[i*quarter_len:(i+1)*quarter_len] for i in range(4)]
        
        min_vals = [np.min(quarter) for quarter in quarters]
        max_vals = [np.max(quarter) for quarter in quarters]
        mean_vals = [np.mean(quarter) for quarter in quarters]
        
        stats[f'deriv_max_1_2_{col}'] = np.max(quarters[1]) - np.max(quarters[0])
        stats[f'deriv_max_3_4_{col}'] = np.max(quarters[3]) - np.max(quarters[2])
        stats[f'deriv_min_1_2_{col}'] = np.min(quarters[1]) - np.min(quarters[0])
        stats[f'deriv_min_3_4_{col}'] = np.min(quarters[3]) - np.min(quarters[2])
        
        # Shannon Entropy for 1-second window
        signal_normalized = signal / (np.sum(signal) + epsilon)
        stats[f'shannon_entropy_{col}'] = -np.sum(signal_normalized * np.log(signal_normalized + epsilon))
        
        # FFT Analysis
        fft_vals = fft(signal)
        fft_abs = np.abs(fft_vals)
        stats[f'fft_mean_{col}'] = np.mean(fft_abs)
        stats[f'fft_std_{col}'] = np.std(fft_abs)
        stats[f'fft_kurtosis_{col}'] = kurtosis(fft_abs)
        stats[f'fft_skewness_{col}'] = skew(fft_abs)

        # Power Spectral Density (PSD)
        freqs, psd = power_spectral_density(signal)
        stats[f'psd_peak_freq_{col}'] = freqs[np.argmax(psd)]
        stats[f'spectral_entropy_{col}'] = spectral_entropy(psd)
        
        # Differential Entropy for the 1-second window
        stats[f'differential_entropy_{col}'] = differential_entropy(signal)

        # Euclidian Distance
        for i in range(4):
            for j in range(i + 1, 4):
                stats[f'euclid_min_{col}_{i+1}_{j+1}'] = euclidean_distance(min_vals[i], min_vals[j])
                stats[f'euclid_max_{col}_{i+1}_{j+1}'] = euclidean_distance(max_vals[i], max_vals[j])
                stats[f'euclid_mean_{col}_{i+1}_{j+1}'] = euclidean_distance(mean_vals[i], mean_vals[j])

    # Log-Covariance Matrix
    log_cov_features = compute_log_cov_features(stats)
    for idx, val in enumerate(log_cov_features):
        stats[f'log_cov_{idx}'] = val
    
    return stats

def process_eeg_data(df, selected_sensors, original_frequency=1000, new_frequency=200):
    df_selected = df.iloc[:, selected_sensors]

    # Shift the signal to ensure all values are non-negative
    shift_value = np.abs(np.min(df_selected.values))
    df_selected = df_selected + shift_value

    # Resample the data to 200Hz
    num_samples = len(df_selected)
    new_num_samples = int(num_samples * new_frequency / original_frequency)
    resampled_data = resample(df_selected, new_num_samples)

    df_resampled = pd.DataFrame(resampled_data, columns=['TP7', 'FP1', 'FP2', 'TP8'])

    window_length = 200  # 1 second window for 200Hz data
    overlap = 100  # 0.5 second overlap
    step = window_length - overlap

    # Apply sliding windows and extract statistics
    statistics = []
    for start in range(0, len(df_resampled) - window_length + 1, step):
        window = df_resampled.iloc[start:start + window_length]
        statistics.append(extract_statistics(window))

    df_statistics = pd.DataFrame(statistics)

    return df_statistics

def load_data_and_process(file_path, selected_sensors=[3, 4, 32, 40]):
    df = pd.read_csv(file_path)
    df_statistics = process_eeg_data(df, selected_sensors)
    return df_statistics

def gather_data(base_dir):
    sessions_labels = {
        '1': [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3], 
        '2': [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        '3': [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    }
    results = []



    for session in tqdm(sorted(os.listdir(base_dir)), desc="Sessions", position=2):
        session_dir = os.path.join(base_dir, session)
        if os.path.isdir(session_dir):
            labels = sessions_labels[session]
            eeg_folders = os.listdir(session_dir)
            for i, eeg_folder in tqdm(enumerate(sorted(eeg_folders)), desc="EEG folders", total=len(eeg_folders), leave=False, position=0):
                eeg_dir = os.path.join(session_dir, eeg_folder)
                if os.path.isdir(eeg_dir):
                    files = os.listdir(eeg_dir)
                    for file in tqdm(sorted(files), desc="Files", total=len(files), leave=False, position=1):
                        if file.endswith('.csv'):
                            file_path = os.path.join(eeg_dir, file)
                            features = load_data_and_process(file_path)
                            results.append((*features.values, labels[i]))

    return results

def shuffle_and_save(results, output_file):
    tqdm.write("Shuffling data...")
    random.shuffle(results)
    tqdm.write("Saving data...")
    data = {
        'features': [result[:-1] for result in results],
        'labels': [result[-1] for result in results]
    }
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    tqdm.write("Data saved to {}".format(output_file))

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    base_dir = '/home/alex/UVT/thesis/csv_files'
    results = gather_data(base_dir)
    shuffle_and_save(results, 'extracted_features.pkl')

