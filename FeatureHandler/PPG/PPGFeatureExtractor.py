import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Common.FeatureExtractorBase import FeaturesExtractorBase
from Common.PublicData import biosignal_feature_maps, biosignal_maps, DevicesEnum
import numpy as np
from biosppy.signals import ppg as biosppy_ppg

class PPGFeaturesExtractor(FeaturesExtractorBase):

    def preprocess(self, data):
        # Placeholder for actual preprocessing logic
        preprocessed_data = data['InternalADC_A13']  # Use 'InternalADC_A13' column for PPG
        return preprocessed_data

    def register_features(self, feature_names):
        for feature in feature_names:
            biosignal_feature_maps[biosignal_maps[DevicesEnum.shimmer.name][1]].append(feature)

    def extract_all_features(self, data):
        ppg_results = biosppy_ppg.ppg(signal=data, sampling_rate=1000.0, show=False)
        ppg_features = [np.mean(ppg_results.heart_rate), np.std(ppg_results.heart_rate)]
        
        names = ['PPG_Mean_Heart_Rate', 'PPG_Std_Heart_Rate']
        self.register_features(names)
        
        return ppg_features

    def extract_features(self, data):
        preprocessed_data = self.preprocess(data)
        ppg_features = self.extract_all_features(preprocessed_data)
        return ppg_features
