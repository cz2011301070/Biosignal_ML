import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Common.FeatureExtractorBase import FeaturesExtractorBase
from Common.PublicData import biosignal_feature_maps, biosignal_maps, DevicesEnum

import numpy as np
from biosppy.signals import emg as biosppy_emg

class EMGFeaturesExtractor(FeaturesExtractorBase):

    def preprocess(self, data):
        # Placeholder for actual preprocessing logic
        preprocessed_data = [data[ch] for ch in ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']]
        return preprocessed_data

    def register_features(self, feature_names):
        for feature in feature_names:
            biosignal_feature_maps[biosignal_maps[DevicesEnum.myo.name][0]].append(feature)

    def extract_all_features(self, data):
        emg_features = []
        names = []
        for i, signal in enumerate(data):
            results = biosppy_emg.emg(signal=signal, sampling_rate=1000.0, show=False)
            features = [np.mean(results.filtered), np.std(results.filtered), len(results.onsets)]
            emg_features.extend(features)
            names.extend([f'EMG_Ch{i+1}_Mean_Filtered', f'EMG_Ch{i+1}_Std_Filtered', f'EMG_Ch{i+1}_Onsets'])
        
        self.register_features(names)
        return emg_features

    def extract_features(self, data):
        preprocessed_data = self.preprocess(data)
        emg_features = self.extract_all_features(preprocessed_data)
        return emg_features
