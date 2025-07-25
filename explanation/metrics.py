import cv2
from features.feature_extraction import extract_fragment_features
from explanation.utils import predict
import numpy as np
import torch
from scipy.stats import kendalltau
from abc import ABC, abstractmethod

class MetricsCalculator(ABC):
    @abstractmethod
    def compute_discoverability(self):
        pass

#Fragment-level explanation evaluation metrics
class MetricsFragmentCalculator(MetricsCalculator):
    def __init__(self, model, original_features, fragments_frame_index):
        self.model = model
        self.features = original_features
        self.fragments_frame_index=fragments_frame_index

    #Compute the discoverability scores
    def compute_discoverability(self, fragments, number_of_fragments):
        results=[]
        result = predict(self.features, self.model)
        #Compute the discoverability scores in a one-by-one manner (mask out one fragment each time)
        #For the number of top fragments (if they exist)
        for s in fragments[:number_of_fragments]:
            #Mask out the current fragment frames with black frames
            masked = [np.zeros((224, 224, 3), dtype=np.float32) for _ in range(len(self.fragments_frame_index[s]))]
            #Extract the deep features
            frame_features=extract_fragment_features(np.array(masked))
            frame_features = torch.Tensor(np.array(frame_features)).view(-1, 1024)
            frame_features = frame_features.to(self.model.linear_1.weight.device)
            new_features = self.features.detach().clone()
            #Replace the new deep features of the masked fragment
            new_features[self.fragments_frame_index[s]]=frame_features
            masked_result=predict(new_features,self.model)
            #Compute the kendall coefficient between the original and the masked fragment scores
            results.append(kendalltau(result, masked_result)[0])
        discoverability1 = results

        #Similarly, compute the discoverability scores in a sequential (batch) manner (mask out the fragments sequentially)
        results = []
        new_features = self.features.detach().clone()
        for s in fragments[:number_of_fragments]:
            masked = [np.zeros((224, 224, 3), dtype=np.float32) for _ in range(len(self.fragments_frame_index[s]))]
            frame_features = extract_fragment_features(np.array(masked))
            frame_features = torch.Tensor(np.array(frame_features)).view(-1, 1024)
            frame_features = frame_features.to(self.model.linear_1.weight.device)
            new_features[self.fragments_frame_index[s]] = frame_features
            masked_result = predict(new_features, self.model)
            results.append(kendalltau(result, masked_result)[0])
        discoverability2 = results

        return [discoverability1,discoverability2]
