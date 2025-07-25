import numpy as np
import sys
from functools import partial
sys.path.append('../')
import torch
import sklearn
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from features.feature_extraction import extract_fragment_features
from explanation.utils import predict
from abc import ABC, abstractmethod

#Get the visual video explanation
class VideoExplanation(object):
    def __init__(self, segments):
        self.segments = segments-1
        self.local_exp = {}

    #Return the visual explanation superimposed onto the keyframe of the fragment
    def get_image_and_mask(self, image, num_features, positive_only=True, negative_only=False, hide_rest=False):

        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")

        #Segmentation Mask
        segments = self.segments
        #Ranked regressor weights for each segment
        exp = self.local_exp[0]
        mask = np.zeros(segments.shape, segments.dtype)

        #If we want for the non-explanation part of the returned image to be black
        if hide_rest:
            #Fill the explanation array with zeros
            temp = np.zeros(image.shape).astype(np.uint8)
        #Otherwise
        else:
            #Copy the keyframe and set the explanation array
            temp = image.copy()
        #If we want to only take segments that positively contribute to the prediction of the label
        if positive_only:
            #Create a ranked array with the most important num_features in desceding order
            fs = [x[0] for x in exp if x[1] > 0][:num_features]
        #If we want to only take the segments that negatively contribute to the prediction of the label
        if negative_only:
            #Create a ranked array with the least important num_features in desceding order
            fs = [x[0] for x in exp if x[1] < 0][:num_features]
        #If one of the above conditions are satisfied
        if positive_only or negative_only:
            #Create the explanation image and the explanation mask
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            #Return the explanation image and mask
            return temp, mask
        #Otherwise mark both the positive and the negative num_features
        else:
            #Initialize the marked positive and negative segment counters
            count_pos = 0
            count_neg = 0
            #For every ranked segment based on the regressor weight
            for f, w in exp:
                #If the weight is positive and the number of marked positive segments has not been reached yet
                if (w > 0 and count_pos < num_features):
                    #Mark the current segment on the mask with 1
                    mask[segments == f] = 1
                    temp[segments == f] = image[segments == f].copy()
                    #Darken the corresponding location of the explanation image so the overlayed color is better visible
                    temp[segments == f] = temp[segments == f] / 4
                    #Overlay green color onto the location of the segment at the explanation image
                    temp[segments == f, 1] = np.max(image)
                    #Advance the positive marked segment counter
                    count_pos += 1
                #If the weight is negative and the number of marked negative segments has not been reached yet
                elif (w < 0 and count_neg < num_features):
                    #Mark the current segment on the mask with 1
                    mask[segments == f] = -1
                    temp[segments == f] = image[segments == f].copy()
                    #Darken the correspoding location of the explanation image so the overlayed color is better visible
                    temp[segments == f] = temp[segments == f] / 4
                    #Overlay red color onto the location of the segment at the explanation image
                    temp[segments == f, 0] = np.max(image)
                    #Advance the negative marked segment counter
                    count_neg += 1
                else:
                    continue
            #Return the explanation image and mask
            return temp, mask


class LimeExplainer(ABC):
    def __init__(self):
        self.l = 0

        #Define the kernel to compute the weights of the perturbations based on their distance from the original frame/video (all segments/fragments are included)
        kernel_width = float(.25)

        def kernel(d, kernel_width):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.kernel_fn = partial(kernel, kernel_width=kernel_width)
        #Initialize a random state to generate random values
        self.random_state = check_random_state(None)  #Return the RandomState singleton used by np.random

    @abstractmethod
    def explain_instances(self):
        pass

    def data_labels(self):
        pass

    #Fit the data to the labels with a linear regressor
    def explain_instance_with_data(self, neighbourhood_data, neighbourhood_labels, distances, segments):
        used_features = [i for i in range(np.max(segments))]
        #Compute the weights of the data based on the pairwise distances of the binary representations of the perturbations
        weights = self.kernel_fn(distances)
        labels_column = neighbourhood_labels[:]

        #Initialize a ridge regressor
        model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)  # Create a regressor model

        #Fit the features (each weight of the regressor corresponds to a segment/fragment)
        _ = model_regressor.fit(neighbourhood_data[:, used_features], labels_column,sample_weight=weights)  #Fit the features

        #Return the scores a sorted zip object, based on the absolute value of the weights, with pairs of segments/fragments and the values of the corresponding weights
        return  sorted(zip(used_features, model_regressor.coef_),
                key=lambda x: np.abs(x[1]), reverse=True)


#Fragment level LIME explainer
class LimeFragmentExplainer(LimeExplainer):
    def __init__(self):
        super().__init__()

    #Generate the explanation for the video fragments
    def explain_instances(self, perturbations, features, fragment_frames_index, model, num_of_fragments):
        #Get the binary data array containing the generated perturbations and their corresponding labels given by the classifier
        data, labels = self.data_labels(perturbations, features, fragment_frames_index, model, num_of_fragments)
        #Compute the pairwise distances of the binary representation of the perturbations with the original
        #(rows containing 0's and 1's, indicating if the fragment in that index is contained or not in the perturbation)
        distances = sklearn.metrics.pairwise_distances(data, data[0].reshape(1, -1), metric='cosine').ravel()

        #Fit the data to the labels with a linear regressor
        #local_exp is a ranked zip object, based on the absolute value of the weights, with pairs of segments/fragments and the values of the corresponding weights
        return self.explain_instance_with_data(data, labels, distances,len(fragment_frames_index))

    #Create the binary data perturbations and compute the corresponding scores
    def data_labels(self, perturbed_num, original_features, frame_indexes, model, num_of_fragments):
        labels = []
        #Set the number of features the same as the number of fragments of the video
        n_features = num_of_fragments
        #Create a random binary data array with 0's and 1's indicating if the fragment in the corresponding index is masked out or not
        #Shape (perturbations, fragments)
        data = self.random_state.randint(0, 2, perturbed_num * n_features).reshape((perturbed_num, n_features))
        # Set the first perturbation with the original video (all fragments included)
        data[0, :] = 1

        #Extract the deep features of a black frame
        black_frame_features = extract_fragment_features(np.expand_dims(np.zeros((224, 224, 3), dtype=np.float32), axis=0))
        black_frame_features = torch.Tensor(np.array(black_frame_features)).view(-1, 1024)
        black_frame_features = black_frame_features.to(model.linear_1.weight.device)
        #For each perturbation (row in data)
        for i in range(len(data)):
            #Mask the frames of the fragments corresponding to the indexes marked with 0's
            zero_indices = np.where(data[i] == 0.0)[0]
            mask = [frame_indexes[j] for j in zero_indices]
            mask = sum(mask, [])
            features = original_features.detach().clone()
            #Replace the fragment features with the new perturbated features
            features[mask] = black_frame_features
            #Compute the scores
            result = predict(features, model)
            #The perturbation score is equal to the average of the scores of all the frames
            perturbation_score = np.mean(result)
            labels.append(perturbation_score)

        return [np.array(data), np.array(labels)]