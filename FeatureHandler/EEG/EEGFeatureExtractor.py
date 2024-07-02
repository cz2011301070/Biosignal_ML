import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Common.FeatureExtractorBase import FeaturesExtractorBase
from Common.PublicData import biosignal_feature_maps, biosignal_maps, DevicesEnum
import numpy as np
from biosppy.signals import eeg as biosppy_eeg

class EEGFeaturesExtractor(FeaturesExtractorBase):

    def preprocess(self, data):
        # Placeholder for actual preprocessing logic
        preprocessed_data = np.column_stack([data[ch] for ch in ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]])
        return preprocessed_data

    def register_features(self, feature_names):
        for feature in feature_names:
            biosignal_feature_maps[biosignal_maps[DevicesEnum.unicorn.name][0]].append(feature)

    def extract_all_features(self, data):
        results = biosppy_eeg.eeg(signal=data, sampling_rate=1000.0, show=False)
        eeg_features = np.concatenate([results.theta.mean(axis=0), results.alpha_low.mean(axis=0), results.alpha_high.mean(axis=0), results.beta.mean(axis=0), results.gamma.mean(axis=0)])
        
        names = ['EEG_Theta_Mean', 'EEG_Alpha_Low_Mean', 'EEG_Alpha_High_Mean', 'EEG_Beta_Mean', 'EEG_Gamma_Mean']
        self.register_features(names)
        
        return eeg_features

    def extract_features(self, data):
        preprocessed_data = self.preprocess(data)
        eeg_features = self.extract_all_features(preprocessed_data)
        return eeg_features
