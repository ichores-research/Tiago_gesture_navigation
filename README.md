
# Bachelor Thesis Repository: Gesture Operation of Humanoid Robot Tiago 

This is a repository to bachelor thesis on working with humanoid robot TIAGo++ using human pose recognition solution from MediaPipe, processing image data with neural networks [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) and [MotionBERT](https://github.com/Walter0807/MotionBERT) with goal to find human in a wokspace around the robot, recognize the direction in which a human is pointing and moving to said point.


## Description of folders in this repository

- [tiago_dual_python_controller](tiago_dual_python_controller): ROS package containing main script [main.py](tiago_dual_python_controller/scripts/main.py) alongside with script to process image by MediaPipe solution [mediapipe_image_server.py](tiago_dual_python_controller/scripts/mediapipe_image_server.py). Also contains images created during testing in simulation


- [alphapose_changed](alphapose_changed): Folder with files from original [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) repository, which were changed to make the data processing repetable.

- [motionbert_changed](motionbert_changed): Folder with files from original [MotionBERT](https://github.com/Walter0807/MotionBERT) repository, which were changed to make the data processing repetable.

- [image_share_service](image_share_service): Folder containing classes used for comunicating accros running scripts via LAN, used for data transfer
