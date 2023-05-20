
# Bachelor Thesis Repository: Gesture Operation of Humanoid Robot Tiago 

This is a repository to bachelor thesis on working with humanoid robot TIAGo++ using human pose recognition solution from MediaPipe, processing image data with neural networks [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) and [MotionBERT](https://github.com/Walter0807/MotionBERT) with goal to find human in a wokspace around the robot, recognize the direction in which a human is pointing and moving to said point.


## Description of folders in this repository

- [tiago_dual_python_controller](tiago_dual_python_controller): ROS package containing main script [main.py](tiago_dual_python_controller/scripts/main.py) alongside with script to process image by MediaPipe solution [mediapipe_image_server.py](tiago_dual_python_controller/scripts/mediapipe_image_server.py). Also contains images created during testing in simulation


- [alphapose_changed](alphapose_changed): Folder with files from original [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) repository, which were changed to make the data processing repetable.

- [motionbert_changed](motionbert_changed): Folder with files from original [MotionBERT](https://github.com/Walter0807/MotionBERT) repository, which were changed to make the data processing repetable.

- [image_share_service](image_share_service): Folder containing classes used for comunicating accros running scripts via LAN, used for data transfer


## Installation and requirements

The requirements to run this project are

- Both Python 2.7 and Python 3.6.9

- Installed AlphaPose conda enviroment (installable from [environment_alphapose.yml](environment_alphapose.yml)) alongside with downloaded repo, Fast Pose model trained on Halpe (26 keypoits) dataset.

- Installed MotionBERT conda enviroment (installable from [environment_motionbert.yml.yml](environment_motionbert.yml)) alongside with downloaded repo, everything required for 3D Pose estimation from this repo

- Ubuntu 18.04 with ROS Melodic and TIAGo++ workspace installed by [this](http://wiki.ros.org/Robots/TIAGo%2B%2B/Tutorials/Installation/InstallUbuntuAndROS) guide

- few additional packages for comunication between running scripts: zmq, cloudpickle, mediapipe (version 0.8.2 for Python 3.6.9)


After the steps above are complete, do following:

- from [alphapose_changed](alphapose_changed) folder copy everything there to original AlphaPose repo and put there [image_share_service](image_share_service) as well 

- from [motionbert_changed](motionbert_changed) folder copy everything there to original AlphaPose repo and put there [image_share_service](image_share_service) as well 
