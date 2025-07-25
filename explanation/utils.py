# -*- coding: utf-8 -*-
import ntpath
import os
import h5py
import numpy as np
import torch
from features.feature_extraction import extract_features
from layers.summarizer import CA_SUM

#Load and return the pretrained CA-SUM model
def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CA_SUM(input_size=1024, output_size=1024, block_size=60).to(device)   #Default parameters
    model.load_state_dict(torch.load(model_path))

    return model

#Get the h5 file that contains the deep features of the sampled video frames
def getH5File(video_path,save_path):

    #If the file already exists and contains only the deep features for the current video (is in the save directory of the video)
    if (os.path.isfile(save_path + "/features.h5")):
        #Set the corresponding path (save path)
        h5_path = save_path + "/features.h5"
    #If the file already exists and contains all of the deep features of the videos (is in the main directory of the dataset)
    elif (os.path.isfile(os.path.dirname(video_path) + "/features.h5")):
        #Set the corresponding path (dataset directory path)
        h5_path = os.path.dirname(video_path) + "/features.h5"
    #Othewise extract the deep features for the current video and save it to the save path (save directory of the video)
    else:
        extract_features(video_path, save_path)
        h5_path = save_path + "/features.h5"

    #Read and return the contents of the h5 file
    return h5py.File(h5_path, "r")

#Load the deep features of the video
def load_data(video_path,model):

    #Get the h5 type file containing the deep features
    hdf = getH5File(video_path,video_path[:-4])

    #Extract the frame features for inference
    video_name = ntpath.basename(video_path)[:-4]
    frame_features = torch.Tensor(np.array(hdf[f"{video_name}/features"])).view(-1, 1024)
    frame_features = frame_features.to(model.linear_1.weight.device)

    #Return the frame features
    return frame_features

#Inference the CA-SUM model and return the output scores
def predict(features, model):

    model.eval()
    with torch.no_grad():
        scores, _ = model(features)
        scores = scores.squeeze(0).cpu().numpy().tolist()

    return scores