import pandas as pd
import numpy as np
from biosppy.signals import eda as biosppy_eda, ppg as biosppy_ppg, eeg as biosppy_eeg, emg as biosppy_emg
import biosppy
from pylsl import StreamInlet, resolve_stream
import threading
import time
import queue
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Common.PublicData import NUM_SAMPLES
def extract_shimmer_features(data):
    eda_signal = data['GSR']
    ppg_signal = data['InternalADC_A13']
    
    # EDA features
    eda_results = biosppy_eda.eda(signal=eda_signal, sampling_rate=1000.0, show=False)
    eda_features = np.array([np.mean(eda_results.amplitudes), np.std(eda_results.amplitudes)])
    #Mean amplitude and standard deviation of skin conductance (EDA amplitude).
    """This function filters the raw EDA signal, detects SCRs, and computes their amplitudes.
It provides:
filtered: Filtered EDA signal.
onsets: Indices of SCR onsets.
peaks: Indices of SCR peaks.
amplitudes: Amplitudes of SCRs."""
    
    # PPG features
    ppg_results = biosppy_ppg.ppg(signal=ppg_signal, sampling_rate=1000.0, show=False)
    ppg_features = np.array([np.mean(ppg_results.heart_rate), np.std(ppg_results.heart_rate)])
    #Mean heart rate and standard deviation of heart rate.
    return np.concatenate([eda_features, ppg_features])
"""Processing: Filters the raw PPG signal and identifies pulse onsets.
Features Extracted:
Filtered Signal: Processed PPG signal.
Pulse Onsets: Indices indicating the start of each pulse.
Heart Rate: Instantaneous heart rate derived from pulse intervals.
Heart Rate Variability: Features related to heart rate variability may also be extracted."""

def extract_emg_features(data):
    emg_signals = [data[ch] for ch in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']]
    emg_features = []
    for signal in emg_signals:
        results = biosppy_emg.emg(signal=signal, sampling_rate=1000.0, show=False)
        features = np.array([np.mean(results.filtered), np.std(results.filtered), len(results.onsets)])
        emg_features.append(features)
    
    return np.concatenate(emg_features)
    #Mean and standard deviation of filtered EMG signals, and the number of onsets detected.
"""Processing: Filters the raw EMG signal and detects muscle activation (onsets).
Features Extracted:
Filtered Signal: Processed EMG signal.
Onsets: Indices indicating muscle activation points.
Statistics: Typically includes mean, standard deviation, and count of muscle activations (onsets)."""

def extract_eeg_features(data):
    eeg_signals = np.column_stack([data[ch] for ch in ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]])
    results = biosppy_eeg.eeg(signal=eeg_signals, sampling_rate=1000.0, show=False)
    eeg_features = np.concatenate([results.theta.mean(axis=0), results.alpha_low.mean(axis=0), results.alpha_high.mean(axis=0), results.beta.mean(axis=0), results.gamma.mean(axis=0)])
    
    return eeg_features
    #Mean power in theta, alpha low, alpha high, beta, and gamma frequency bands.
"""Applies Common Average Reference (CAR) to the raw EEG signals, then extracts frequency-specific power and phase locking values (PLV).
Features Extracted:
Filtered Signals: Processed EEG signals.
Frequency Bands: Average power in various frequency bands (e.g., theta, alpha, beta, gamma).
Phase Locking Values (PLV): Measures of synchrony between EEG signals across different electrode pairs."""

# Main processing function
def process_data(shimmer_queue, emg_queue, eeg_queue, participant_id, condition_id):
    combined_features = []  # Combined features from all streams
    labels = []  # Placeholder for labels, adjust as necessary
    data_folder = f"data_{participant_id}_{condition_id}"
    os.makedirs(data_folder, exist_ok=True)
    
    while True:
        # Get samples from all queues
        shimmer_data = shimmer_queue.get()
        emg_data = emg_queue.get()
        eeg_data = eeg_queue.get()
        
        # Check if any queue has received a termination signal
        if shimmer_data is None or emg_data is None or eeg_data is None:
            break
        
        # Extract features from each stream
        shimmer_features = extract_shimmer_features(shimmer_data)
        emg_features = extract_emg_features(emg_data)
        eeg_features = extract_eeg_features(eeg_data)
        
        # Combine features into a single feature vector
        combined_features.append(np.concatenate([shimmer_features, emg_features, eeg_features]))
        
        # Train the model with the combined feature vector (example)
        if len(labels) >= len(combined_features):
            X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Model accuracy: {accuracy * 100:.2f}%')
            # Reset combined_features for the next training iteration
            combined_features = []

# Define a queue for each device data stream
shimmer_queue = queue.Queue()
unicorn_queue = queue.Queue()
myo_queue = queue.Queue()

# Data reception function for each device
def receive_data(device_type, device_queue):
    print(f"Looking for a {device_type} stream...")
    streams = resolve_stream('name', device_type)
    inlet = StreamInlet(streams[0])
    
    while True:
        sample, timestamp = inlet.pull_sample()
        device_queue.put([timestamp] + sample)

# Create threads for data reception
shimmer_thread = threading.Thread(target=receive_data, args=('shimmer', shimmer_queue))
emg_thread = threading.Thread(target=receive_data, args=('myo', myo_queue))
eeg_thread = threading.Thread(target=receive_data, args=('unicorn', unicorn_queue))

# Create thread for data processing

processing_thread = threading.Thread(target=process_data, args=(shimmer_queue, myo_queue, unicorn_queue, participant_id, condition_id))

# Start the threads
shimmer_thread.start()
emg_thread.start()
eeg_thread.start()
processing_thread.start()

# Join the threads to the main thread
shimmer_thread.join()
emg_thread.join()
eeg_thread.join()
processing_thread.join()
