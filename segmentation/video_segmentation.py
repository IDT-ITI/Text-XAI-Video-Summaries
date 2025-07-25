import os
import subprocess
import cv2
import numpy


#Path of the root directory of the TransNetV2 project relative to the parent directory of the current project
transnet_root_dir_path="./TransNetV2"
#Path of the virtual environment directory of the TransNetV2 project relative to the root directory of the TransNetV2 project
transnet_venv_dir_path="./.venv"


#Get the video frame count returned from the cv2 video capture
def getCVFrameCount(vidcap):
    count=0

    while True:
        success, _ = vidcap.read()
        if (not(success)): break
        count += 1

    return count

#Get the TransNetV2 shots for the video
def getTransNetShots(video_path,last_frame):

    #Set the required paths to call the subprocess to run TransNetV2
    python_executable = "../"+transnet_venv_dir_path.strip("/")+"/bin/python"
    command_and_args = [
        python_executable,
        "transnetv2.py",
        os.path.abspath(video_path)
    ]

    root_dir = '/'.join(os.path.abspath(__file__).split('/')[:-3])+ "/"+ transnet_root_dir_path.strip("/")
    working_dir = root_dir + "/inference"
    #Set PYTHONPATH to include the Python package path
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir

    #Call the TransNetV2 subprocess
    p = subprocess.Popen(command_and_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_dir, env=env)
    p.wait()
    stdout, stderr = p.communicate()

    #Load the txt returned from TransNetV2 that contains the video shots
    shots = numpy.loadtxt(video_path+".scenes.txt", delimiter=' ', dtype=int).reshape(-1,2)

    #Remove the returned TransNetV2 files as they are no longer needed
    os.remove(video_path + ".scenes.txt")
    os.remove(video_path + ".predictions.txt")

    #Substitute the last frame number with the frame count returned from the cv2 video capture
    #(There are cases where the video capture doesn't read some of the last frames so we omit them from the frame count)
    shots[-1][-1]=last_frame

    #Return the shots
    return list(shots)

#Get the video shots
def getShots(video_path):
    #Set the optical flow subshots to false by default
    opt_shots = False

    shot_path=video_path[:-4]

    #If the optical flow subshots file exists
    if(os.path.isfile(shot_path + "/opt_shots.txt")):
        #Set the optical flow subshots to true
        opt_shots=True

    #When we have optical flow subshots
    if(opt_shots):
        try:
            #Try reading the file as it was returned by the program, change the txt formatting and return the shots
            shots = numpy.loadtxt(shot_path + "/opt_shots.txt", delimiter=' ', dtype=int,ndmin=2)[:, :2].reshape(-1, 2)
            numpy.savetxt(shot_path + "/opt_shots.txt", shots, "%d,%d")
            return list(shots)
        #Otherwise the file has already been opened and the format has already been changed
        except:
            #Read the txt file and return the shots
            shots = numpy.loadtxt(shot_path + "/opt_shots.txt", delimiter=',', dtype=int).reshape(-1,2)
            return list(shots)
    else:
        #If TransNetV2 has already computed the shots
        if (os.path.isfile(shot_path + "/shots.txt")):
            #Read the txt file containing them and return the shots
            shots = numpy.loadtxt(shot_path + "/shots.txt", delimiter=',', dtype=int).reshape(-1, 2)
            return list(shots)
        #Othewise call TransNetV2
        else:
            vidcap = cv2.VideoCapture(video_path)
            #Video frame count returned from the cv2 video capture
            last_frame=getCVFrameCount(vidcap)
            #Call TransNetV2
            shots = getTransNetShots(video_path,last_frame)
            #If the shot save path doesn't exist, create it
            if(not os.path.exists(shot_path)):
                os.mkdir(shot_path)
            #Save the shots into a txt file
            numpy.savetxt(shot_path + "/shots.txt", shots, "%d,%d")
            #Return the shots
            return list(shots)

if __name__ == "__main__":

    video_name="video_1"
    dataset="TVSum"

    video_path = '../data/' + dataset + '/' + video_name + '.mp4'

    shots=getShots(video_path)

    print("Shots:")
    for s in shots:
        print(s[0],s[1])

