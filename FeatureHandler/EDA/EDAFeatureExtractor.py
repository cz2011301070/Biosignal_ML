import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Common.FeatureExtractorBase import FeaturesExtractorBase
from Common.PublicData import biosignal_feature_maps, biosignal_maps, DevicesEnum
import numpy as np
from biosppy.signals import eda as biosppy_eda

class EDAFeaturesExtractor(FeaturesExtractorBase):

    def preprocess(self, data):
        # Placeholder for actual preprocessing logic
        preprocessed_data = data['GSR']  # Use 'GSR' column for EDA
        return preprocessed_data

    def register_features(self, feature_names):
        for feature in feature_names:
            biosignal_feature_maps[biosignal_maps[DevicesEnum.shimmer.name][0]].append(feature)

    def extract_all_features(self, data):
        eda_results = biosppy_eda.eda(signal=data, sampling_rate=1000.0, show=False)
        eda_features = [np.mean(eda_results.amplitudes), np.std(eda_results.amplitudes)]
        
        names = ['EDA_Mean_Amplitude', 'EDA_Std_Amplitude']
        self.register_features(names)
        
        return eda_features

    def extract_features(self, data):
        preprocessed_data = self.preprocess(data)
        eda_features = self.extract_all_features(preprocessed_data)
        return eda_features
