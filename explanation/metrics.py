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

        #Similarly, compute the discoverability scores in a sequential (batch) manner (mask out the fragments sequentially)
        results = []
        result = predict(self.features, self.model)
        new_features = self.features.detach().clone()
        for s in fragments[:number_of_fragments]:
            masked = [np.zeros((224, 224, 3), dtype=np.float32) for _ in range(len(self.fragments_frame_index[s]))]
            frame_features = extract_fragment_features(np.array(masked))
            frame_features = torch.Tensor(np.array(frame_features)).view(-1, 1024)
            frame_features = frame_features.to(self.model.linear_1.weight.device)
            new_features[self.fragments_frame_index[s]] = frame_features
            masked_result = predict(new_features, self.model)
            results.append(kendalltau(result, masked_result)[0])
        discoverability = results

        return [discoverability]
