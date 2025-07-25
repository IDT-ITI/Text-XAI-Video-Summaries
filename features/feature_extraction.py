import ntpath
import os
import h5py
import torch
import cv2
import numpy as np
import torchvision
from torchvision import transforms
import math

#Extract the sampled video frames
def load_video(video, desired_size=(224, 224)):

    vidcap = cv2.VideoCapture(video)
    frames = []
    count = 0
    #Sample every 15 frames
    interval = 15
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        #If all frames have been read then break
        if ret == False:
            break
        #If the index of the current frame is multiple of the interval 15
        if (count % interval) == 0:
            #Sample the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Resize it if needed
            if desired_size != 0:
                frame = cv2.resize(frame, desired_size)
            frames.append(frame)
        count += 1

    n_frames = count
    vidcap.release()
    #Return the frames and the count number
    return np.array(frames), n_frames

#Get the deep features from the GoogleNet model
def google_features(frames, batch_size):
    #Load the model
    model = torchvision.models.googlenet(pretrained=True)
    #Set the output of the model to the output of the pool5 layer
    model = torch.nn.Sequential(*list(model.children())[:-2])

    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #Transform the frames to pass them to the model
    video_tensor = []
    for frame in frames:
        video_tensor.append(preprocess(frame))
    video_tensor = torch.stack(video_tensor)

    if torch.cuda.is_available():
        model.to('cuda')
        video_tensor = video_tensor.to('cuda')
    model.eval()

    features = []
    num_frames = video_tensor.shape[0]
    num_iter = math.ceil(num_frames / batch_size)

    #Create frame batches and pass them to the model to get the output features
    for iter in range(num_iter):
        batch_frames = video_tensor[iter * batch_size:min(num_frames,(iter + 1) * batch_size)]

        with torch.no_grad():
            output = model(batch_frames).squeeze().cpu()

        features.append(np.array(output).reshape(-1,1024))

    features = np.concatenate(features)
    #Return the features
    return features

#Extract the features of all the frames of a video and save them into a h5 file
def extract_features(video_path, data_path, batch_size=64, h5_name='features.h5'):
    video_name = ntpath.basename(video_path)
    #Load the video frames
    frames, n_frames = load_video(video_path)
    #Get their deep features
    features = google_features(frames,batch_size)

    #Create and open an h5 file to save the features of the video frames ('a' mode to append features for multiple videos)
    hdf = h5py.File(os.path.join(data_path, h5_name), 'a')
    hdf.create_dataset(video_name[:-4] + '/features', data=features)
    hdf.create_dataset(video_name[:-4] + '/n_frames', data=n_frames)

    hdf.close()

#Extract the features of the frames of a fragment
def extract_fragment_features(frames,batch_size=64):
    # Get the deep features of the fragment frames and return them
    features = google_features(frames, batch_size)
    return features

if __name__ == '__main__':
    for i in range(25):
        extract_features('../data/SumMe/video_'+str(i+1)+'.mp4','../data/SumMe/')
    for i in range(50):
        extract_features('../data/TVSum/video_'+str(i+1)+'.mp4','../data/TVSum/')